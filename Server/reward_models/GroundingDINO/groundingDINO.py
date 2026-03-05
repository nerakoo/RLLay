import os
import cv2
import numpy as np
from PIL import Image
import groundingdino.datasets.transforms as T
import torch
from torchvision.ops import nms
from groundingdino.util.inference import load_model, load_image, predict, annotate

class GroundingDINOEvaluator:
    def __init__(self,
                 model_config_path: str,
                 model_ckpt_path: str,
                 box_threshold: float = 0.3,
                 text_threshold: float = 0.3):

        self.model_config_path = model_config_path
        self.model_ckpt_path = model_ckpt_path
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.model = load_model(self.model_config_path, self.model_ckpt_path)

    def load_input_image(self, image_input):
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        if isinstance(image_input, str) and os.path.exists(image_input):
            pil_img = Image.open(image_input).convert("RGB")
            image_source = np.array(pil_img)
            image_transformed, _ = transform(pil_img, None)
            return image_source, image_transformed
        elif isinstance(image_input, np.ndarray):
            image_source = image_input.copy()
            pil_img = Image.fromarray(image_source)
            image_transformed, _ = transform(pil_img, None)
            return image_source, image_transformed
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
            image_source = np.array(pil_img)
            image_transformed, _ = transform(pil_img, None)
            return image_source, image_transformed
        else:
            raise ValueError("Unrecognized image input format")

    def compute_iou(self, boxA, boxB):
        boxA = self.fix_box_order(boxA)
        boxB = self.fix_box_order(boxB)
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def get_box_center(self, box):
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return (cx, cy)

    def compute_distance(self, pointA, pointB):
        return np.sqrt((pointA[0] - pointB[0]) ** 2 + (pointA[1] - pointB[1]) ** 2)

    def normalize_to_absolute(self, box, img_width, img_height):
        cx, cy, w, h = box
        cx_abs = cx * img_width
        cy_abs = cy * img_height
        w_abs = w * img_width
        h_abs = h * img_height
        x0 = cx_abs - w_abs / 2
        y0 = cy_abs - h_abs / 2
        x1 = cx_abs + w_abs / 2
        y1 = cy_abs + h_abs / 2
        return [x0, y0, x1, y1]

    def fix_box_order(self, box):
        x1, y1, x2, y2 = box
        return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

    def apply_nms(self, boxes, scores, iou_threshold=0.5):
        if len(boxes) == 0:
            return []

        # Convert to torch tensors
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)

        # Apply NMS using torchvision
        keep_indices = nms(boxes_tensor, scores_tensor, iou_threshold)

        return keep_indices.tolist()

    def detect_with_groundingDINO(self, image, text_prompt):
        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold
        )
        return boxes, logits, phrases

    def evaluate_image(self, image_input, meta: dict, save_annotated: str = None, nms_threshold: float = 0.5):
        image_source, image = self.load_input_image(image_input)
        img_h, img_w = image_source.shape[:2]

        annotations = meta.get("annotations", [])
        results = []

        for ann in annotations:
            prompt = ann["prompt"]
            boxes, logits, phrases = self.detect_with_groundingDINO(image, prompt)

            # Apply NMS to filter overlapping boxes
            if len(boxes) > 0:
                # Convert boxes to absolute coordinates for NMS
                boxes_abs = [self.normalize_to_absolute(box, img_w, img_h) for box in boxes]
                boxes_abs = [self.fix_box_order(box) for box in boxes_abs]

                # Apply NMS
                keep_indices = self.apply_nms(boxes_abs, logits, iou_threshold=nms_threshold)

                # Filter boxes, logits, and phrases using NMS results
                boxes = [boxes[i] for i in keep_indices]
                logits = [logits[i] for i in keep_indices]
                phrases = [phrases[i] for i in keep_indices]

            if save_annotated is not None:
                annotated_filename = f"{save_annotated}_{prompt.replace(' ', '_')}.jpg"
                annotated_frame = annotate(
                    image_source=image_source,
                    boxes=boxes,
                    logits=logits,
                    phrases=phrases
                )
                cv2.imwrite(annotated_filename, annotated_frame)

            candidate_indices = list(range(len(phrases)))

            if len(candidate_indices) == 0:
                results.append({
                    "prompt": prompt,
                    "gt_bbox": ann.get("bbox", None),
                    "pred_bbox_for_iou": None,
                    "iou": None,
                    "gt_point": ann.get("point", None),
                    "pred_bbox_for_point": None,
                    "center_distance": None
                })
                continue

            best_iou = -1
            best_bbox_for_iou = None
            gt_bbox = ann.get("bbox", None)
            if gt_bbox is not None:
                for i in candidate_indices:
                    pred_box_norm = boxes[i]
                    pred_box_abs = self.normalize_to_absolute(pred_box_norm, img_w, img_h)
                    pred_box_abs = self.fix_box_order(pred_box_abs)
                    cur_iou = self.compute_iou(gt_bbox, pred_box_abs)
                    if cur_iou > best_iou:
                        best_iou = cur_iou
                        best_bbox_for_iou = pred_box_abs

            best_iou = best_iou if isinstance(best_iou, torch.Tensor) else torch.tensor(best_iou)
            best_distance = float('inf')
            best_bbox_for_point = None
            gt_point = ann.get("point", None)
            if gt_point is not None:
                for i in candidate_indices:
                    pred_box_norm = boxes[i]
                    pred_box_abs = self.normalize_to_absolute(pred_box_norm, img_w, img_h)
                    pred_box_abs = self.fix_box_order(pred_box_abs)
                    pred_center = self.get_box_center(pred_box_abs)
                    cur_distance = self.compute_distance(pred_center, gt_point)
                    if cur_distance < best_distance:
                        best_distance = cur_distance
                        best_bbox_for_point = pred_box_abs

            result_item = {
                "prompt": prompt,
                "gt_bbox": gt_bbox,
                "pred_bbox_for_iou": [float(x) for x in best_bbox_for_iou] if best_bbox_for_iou is not None else None,
                "iou": best_iou if best_bbox_for_iou is not None else None,
                "gt_point": gt_point,
                "pred_bbox_for_point": [float(x) for x in
                                        best_bbox_for_point] if best_bbox_for_point is not None else None,
                "center_distance": best_distance if best_bbox_for_point is not None else None
            }
            results.append(result_item)

        return results

    def detect_and_draw(self, image_input, meta: dict, save_path: str):
        image_source, image = self.load_input_image(image_input)
        image_source = image_source.copy()

        annotations = meta.get("annotations", [])
        prompt_list = [ann["prompt"] for ann in annotations]
        combined_prompt = " . ".join(prompt_list) + " ."

        boxes, logits, phrases = self.detect_with_groundingDINO(image, combined_prompt)

        annotated_frame = annotate(
            image_source=image_source,
            boxes=boxes,
            logits=logits,
            phrases=phrases
        )

        cv2.imwrite(save_path, annotated_frame)
        print(f"The test results have been saved to: {save_path}")
        return boxes, logits, phrases

    def draw_meta_boxes(self, image_input, meta: dict, save_path: str):
        image_source, image = self.load_input_image(image_input)
        image_source = image_source.copy()

        for ann in meta.get("annotations", []):
            if "bbox" in ann and ann["bbox"] is not None:
                bbox = ann["bbox"]
                cv2.rectangle(image_source,
                              (int(bbox[0]), int(bbox[1])),
                              (int(bbox[2]), int(bbox[3])),
                              color=(0, 255, 0),
                              thickness=4)
                cv2.putText(image_source,
                            ann.get("prompt", ""),
                            (int(bbox[0]), int(bbox[1]) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2)

        cv2.imwrite(save_path, image_source)
        print(f"The Meta annotation results have been saved to: {save_path}")

    def compute_reward(self, image_input, meta):
        results = self.evaluate_image(image_input, meta)
        iou_values = [item.get("iou", 0.0) for item in results if item.get("iou") is not None]
        overall_reward = np.mean(iou_values) if iou_values else 0.0
        return overall_reward