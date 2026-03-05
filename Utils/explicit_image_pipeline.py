import os, sys
import torch
# from CreatiLayout.utils.bbox_visualization import bbox_visualization, scale_boxes
from PIL import Image
from LayoutDPO.src.models.transformer_sd3_SiamLayout import SiamLayoutSD3Transformer2DModel
from LayoutDPO.src.pipeline.pipeline_sd3_CreatiLayout import CreatiLayoutSD3Pipeline
import json

from benchmark.__init__ import LayoutDPO_reward

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cache_dir = "path_to_sd3_cache_dir"

def transform_data(global_caption, region_caption_list, region_bboxes_list):
    if len(region_caption_list) != len(region_bboxes_list):
        raise ValueError("region_caption_list and region_bboxes_list must be of the same size.")

    meta = {}
    meta["global_caption"] = global_caption
    meta["image_info"] = {"width": 1024, "height": 1024}

    width = meta["image_info"]["width"]
    height = meta["image_info"]["height"]

    annotations = []
    for caption, bbox in zip(region_caption_list, region_bboxes_list):
        abs_bbox = [bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height]
        annotations.append({
            "prompt": caption,
            "bbox": abs_bbox
        })

    meta["annotations"] = annotations

    return meta

def get_datas(json_file="path_to_filtered_data.json"):
    with open(json_file, 'r', encoding='utf-8') as f:
        datas = json.load(f)

    processed_data = []

    for idx, item in enumerate(datas):
        metadata = item.get("metadata", {})
        image_info = metadata.get("image_info", {})
        width = image_info.get("width", 1)
        height = image_info.get("height", 1)

        global_caption_text = metadata.get("global_caption", "")
        global_caption = [global_caption_text]

        bbox_info_list = metadata.get("bbox_info", [])
        region_caption_list = []
        region_bboxes_list = []

        for bbox_item in bbox_info_list:
            detail = bbox_item.get("description", "")
            region_caption_list.append(detail)

            bbox = bbox_item.get("bbox", [])
            if len(bbox) == 4:
                x_min = bbox[0] / width
                y_min = bbox[1] / height
                x_max = bbox[2] / width
                y_max = bbox[3] / height
                region_bboxes_list.append([x_min, y_min, x_max, y_max])

        filename = f"{idx:06d}.jpg"

        processed_data.append({
            "global_caption": global_caption,
            "region_caption_list": region_caption_list,
            "region_bboxes_list": region_bboxes_list,
            "filename": filename
        })

    return processed_data

if __name__ == "__main__":
    processed_data = get_datas()

    model_path = "stabilityai/stable-diffusion-3-medium-diffusers"
    ckpt_path = "HuiZhang0812/CreatiLayout"
    transformer_additional_kwargs = dict(attention_type="layout", strict=True)
    transformer = SiamLayoutSD3Transformer2DModel.from_pretrained(
        ckpt_path, subfolder="SiamLayout_SD3", cache_dir=cache_dir, torch_dtype=torch.float16, **transformer_additional_kwargs)
    pipe = CreatiLayoutSD3Pipeline.from_pretrained(model_path, cache_dir=cache_dir, transformer=transformer, torch_dtype=torch.float16)
    pipe = pipe.to(device)

    seed = 42
    batch_size = 1
    num_inference_steps = 50
    guidance_scale = 7.5
    height = 1024
    width = 1024

    save_root = "CreatiLayout/output"
    img_save_root = os.path.join(save_root, "images")
    os.makedirs(img_save_root, exist_ok=True)
    img_with_layout_save_root = os.path.join(save_root, "images_with_layout")
    os.makedirs(img_with_layout_save_root, exist_ok=True)

    for i in range(300):
        i = i + 1
        global_caption = processed_data[i]["global_caption"]
        region_caption_list = processed_data[i]["region_caption_list"]
        region_bboxes_list = processed_data[i]["region_bboxes_list"]
        filename = processed_data[i]["filename"]

        print("global_caption:", global_caption)
        print("region_caption_list:", region_caption_list)
        print("region_bboxes_list:", region_bboxes_list)
        print("filename:", filename)

        meta = transform_data(global_caption, region_caption_list, region_bboxes_list)

        with torch.no_grad():
            images = pipe(prompt=global_caption * batch_size,
                          generator=torch.Generator(device=device).manual_seed(seed),
                          num_inference_steps=num_inference_steps,
                          guidance_scale=guidance_scale,
                          bbox_phrases=region_caption_list,
                          bbox_raw=region_bboxes_list,
                          height=height,
                          width=width
                          )
        images = images.images

        reward = LayoutDPO_reward(images[0], meta=meta)
        print("reward:", reward)
        reward = LayoutDPO_reward(images[1], meta=meta)
        print("reward:", reward)
        reward = LayoutDPO_reward(images[2], meta=meta)
        print("reward:", reward)
        reward = LayoutDPO_reward(images[3], meta=meta)
        print("reward:", reward)

        # for j, image in enumerate(images):
        #     image.save(os.path.join(img_save_root, f"{filename}_{j}.png"))
        #
        #     img_with_layout_save_name = os.path.join(img_with_layout_save_root, f"{filename}_{j}.png")
        #
        #     white_image = Image.new('RGB', (width, height), color='rgb(256,256,256)')
        #     show_input = {"boxes": scale_boxes(region_bboxes_list, width, height), "labels": region_caption_list}
        #
        #     bbox_visualization_img = bbox_visualization(white_image, show_input)
        #     image_with_bbox = bbox_visualization(image, show_input)
        #
        #     total_width = width * 2
        #     total_height = height
        #
        #     new_image = Image.new('RGB', (total_width, total_height))
        #     new_image.paste(bbox_visualization_img, (0, 0))
        #     new_image.paste(image_with_bbox, (width, 0))
        #     new_image.save(img_with_layout_save_name)
