import warnings
import torch.nn as nn
import torch
from datasets import load_dataset
from diffusers.models.activations import FP32SiLU
import json
import random
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

import torch
from typing import List, Optional, Tuple, Union

def transform_meta(prompt_metadatas):
    md = prompt_metadatas.get("metadata", {})
    global_caption = md.get("global_caption", "")
    global_caption = [global_caption]

    bbox_info = md.get("bbox_info", [])
    region_caption_list = []
    region_bboxes_list = []

    image_info = md.get("image_info", {})
    width = image_info.get("width", 1)
    height = image_info.get("height", 1)

    for item in bbox_info:
        desc = item.get("description", "")
        region_caption_list.append(desc)

        bbox = item.get("bbox", [])
        if len(bbox) == 4:
            x_min, y_min, x_max, y_max = bbox
            region_bboxes_list.append([
                x_min / width,
                y_min / height,
                x_max / width,
                y_max / height,
            ])

    return global_caption, region_caption_list, region_bboxes_list

def transform_iter_meta(prompt_metadatas):
    global_caption = []
    region_caption_list = []
    region_bboxes_list = []

    for metadata in prompt_metadatas:
        md = metadata.get("metadata", metadata) if isinstance(metadata, dict) else {}
        md = md or {}

        gc = md.get("global_caption", "")
        if isinstance(gc, (list, tuple)):
            gc = " ".join(str(x) for x in gc)
        elif not isinstance(gc, str):
            gc = str(gc)
        global_caption.append(gc)

        image_info = md.get("image_info", {}) or {}
        width = float(image_info.get("width", 1)) or 1.0
        height = float(image_info.get("height", 1)) or 1.0

        phrases = []
        boxes = []
        bbox_info = md.get("bbox_info", []) or []
        for item in bbox_info:
            if not isinstance(item, dict):
                continue

            desc = item.get("description", "")
            if not isinstance(desc, str):
                desc = str(desc)

            bbox = item.get("bbox", [])
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                x_min, y_min, x_max, y_max = bbox

                nx1 = float(x_min) / width
                ny1 = float(y_min) / height
                nx2 = float(x_max) / width
                ny2 = float(y_max) / height

                x1, x2 = (nx1, nx2) if nx1 <= nx2 else (nx2, nx1)
                y1, y2 = (ny1, ny2) if ny1 <= ny2 else (ny2, ny1)

                def clamp01(v):
                    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)
                x1, y1, x2, y2 = map(clamp01, (x1, y1, x2, y2))

                phrases.append(desc)
                boxes.append([x1, y1, x2, y2])

        region_caption_list.append(phrases)   
        region_bboxes_list.append(boxes)     

    return global_caption, region_caption_list, region_bboxes_list

def kto_get_meta(global_caption, region_caption_list, region_bboxes_list):
    if len(region_caption_list) != len(region_bboxes_list):
        raise ValueError("region_caption_list 和 region_bboxes_list 的长度必须一致。")

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

def get_meta(global_caption, region_caption_list, region_bboxes_list, *, width=1024, height=1024):
    def _clamp(v, lo, hi):
        return lo if v < lo else (hi if v > hi else v)

    def _one_sample_meta(gc: str, rc_list, rb_list):
        if len(rc_list) != len(rb_list):
            raise ValueError("The lengths of region_caption_list and region_bboxes_list must be the same.")
        meta = {
            "global_caption": gc if isinstance(gc, str) else str(gc),
            "image_info": {"width": width, "height": height},
            "annotations": []
        }
        for cap, bbox in zip(rc_list, rb_list):
            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                continue
            x1, y1, x2, y2 = bbox
            ax1, ay1 = float(x1) * width,  float(y1) * height
            ax2, ay2 = float(x2) * width,  float(y2) * height

            if ax1 > ax2: ax1, ax2 = ax2, ax1
            if ay1 > ay2: ay1, ay2 = ay2, ay1

            ax1 = _clamp(ax1, 0.0, float(width))
            ay1 = _clamp(ay1, 0.0, float(height))
            ax2 = _clamp(ax2, 0.0, float(width))
            ay2 = _clamp(ay2, 0.0, float(height))
            meta["annotations"].append({
                "prompt": cap if isinstance(cap, str) else str(cap),
                "bbox": [ax1, ay1, ax2, ay2],
            })
        return meta

    is_batch = isinstance(global_caption, (list, tuple))
    if is_batch:
        if not (isinstance(region_caption_list, (list, tuple)) and isinstance(region_bboxes_list, (list, tuple))):
            raise ValueError("In batch mode, region_caption_list and region_bboxes_list should be of type List[...]. 。")
        if len(global_caption) != len(region_caption_list) or len(global_caption) != len(region_bboxes_list):
            raise ValueError("The lengths of the three lists must be the same.")

        metas = []
        for gc, rc_list, rb_list in zip(global_caption, region_caption_list, region_bboxes_list):
            rc_list = rc_list if isinstance(rc_list, (list, tuple)) else []
            rb_list = rb_list if isinstance(rb_list, (list, tuple)) else []
            metas.append(_one_sample_meta(gc, rc_list, rb_list))
        return metas
    else:
        return _one_sample_meta(global_caption, region_caption_list, region_bboxes_list)

def annotate_and_save(
    images,
    region_bboxes_list,
    region_caption_list,
    save_dir,
    epoch: int,
    i: int,
):
    os.makedirs(save_dir, exist_ok=True)

    pil_images = []
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu()
        for img in images:
            if img.dtype == torch.bfloat16:
                img = img.to(torch.float32)
            arr = img.numpy().transpose(1, 2, 0)
            arr = np.clip(arr, 0, 1)
            arr = (arr * 255).round().astype(np.uint8)
            pil_images.append(Image.fromarray(arr))
    else:
        pil_images = list(images)

    try:
        font = ImageFont.truetype("arial.ttf", size=14)
    except IOError:
        font = ImageFont.load_default()

    for idx, pil in enumerate(pil_images):
        draw = ImageDraw.Draw(pil)
        W, H = pil.size

        for (x0n, y0n, x1n, y1n), label in zip(region_bboxes_list, region_caption_list):
            x0, y0 = x0n * W, y0n * H
            x1, y1 = x1n * W, y1n * H
            if x1 > x0 and y1 > y0:
                draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
                text_bbox = draw.textbbox((x0, y0), label, font=font)
                tw = text_bbox[2] - text_bbox[0]
                th = text_bbox[3] - text_bbox[1]
                draw.rectangle([x0, y0 - th, x0 + tw, y0], fill=(255,255,255,200))
                draw.text((x0, y0 - th), label, fill="red", font=font)

        out_path = os.path.join(save_dir, f"{epoch}_{i}_{idx}.png")
        pil.save(out_path)

def Hicoinput(
    global_caption: list[str],
    region_caption_list: list[str],
    region_bboxes_list: list[list[float]],
    img_size: int = 512,
):
    caption = global_caption[0] if isinstance(global_caption, list) else global_caption
    layo_prompt = [caption] + region_caption_list
    prompt = global_caption

    blank = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    cond_images = [blank]

    for box in region_bboxes_list:
        x1, y1, x2, y2 = [
            int(round(c * img_size))
            for c in box
        ]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, img_size), min(y2, img_size)

        mask = np.zeros_like(blank)
        mask[y1:y2, x1:x2] = 255
        cond_images.append(mask)

    list_cond_image_pil = [Image.fromarray(arr).convert("RGB") for arr in cond_images]

    return prompt, layo_prompt, list_cond_image_pil

def annotate_and_save_with_rank(
    images,
    region_bboxes_list: List[List[List[float]]],  
    region_caption_list: List[List[str]],       
    save_dir: str,
    epoch: int,
    i: int,
    k: Optional[int] = None,
):
    if k is not None:
        save_root = os.path.join(save_dir, f"rank_{k:02d}", f"epoch_{epoch:04d}")
    else:
        save_root = os.path.join(save_dir, f"epoch_{epoch:04d}")
    os.makedirs(save_root, exist_ok=True)

    pil_images = []
    if isinstance(images, torch.Tensor):
        imgs = images.detach().cpu()
        for img in imgs:
            if img.dtype in (torch.bfloat16, torch.float16):
                img = img.to(torch.float32)
            arr = img.numpy()
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = arr.transpose(1, 2, 0)
            arr = np.clip(arr, 0, 1)
            arr = (arr * 255).round().astype(np.uint8)
            pil_images.append(Image.fromarray(arr))
    else:
        for img in images:
            if isinstance(img, Image.Image):
                pil_images.append(img)
            else:
                arr = np.asarray(img)
                if arr.ndim == 3 and arr.shape[0] in (1, 3):  
                    arr = arr.transpose(1, 2, 0)
                if arr.dtype != np.uint8:
                    if issubclass(arr.dtype.type, np.floating):
                        arr = np.clip(arr, 0, 1)
                        arr = (arr * 255).round().astype(np.uint8)
                    else:
                        arr = arr.astype(np.uint8)
                pil_images.append(Image.fromarray(arr))

    B = len(pil_images)

    if not (len(region_bboxes_list) == len(region_caption_list) == B):
        raise ValueError(
            f"Batch size mismatch: images={B}, "
            f"region_bboxes_list={len(region_bboxes_list)}, "
            f"region_caption_list={len(region_caption_list)}"
        )

    try:
        font = ImageFont.truetype("arial.ttf", size=14)
    except IOError:
        font = ImageFont.load_default()

    def clamp01(v: float) -> float:
        return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

    for idx in range(B):
        pil = pil_images[idx]
        draw = ImageDraw.Draw(pil)
        W, H = pil.size

        bboxes_i = region_bboxes_list[idx] or []
        labels_i = region_caption_list[idx] or []

        L = min(len(bboxes_i), len(labels_i))
        for j in range(L):
            bbox = bboxes_i[j]
            label = str(labels_i[j])

            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                continue
            x0n, y0n, x1n, y1n = map(float, bbox)

            x0, x1 = sorted([clamp01(x0n), clamp01(x1n)])
            y0, y1 = sorted([clamp01(y0n), clamp01(y1n)])
            x0p, y0p, x1p, y1p = x0 * W, y0 * H, x1 * W, y1 * H

            if x1p > x0p and y1p > y0p:
                draw.rectangle([x0p, y0p, x1p, y1p], outline="red", width=2)
                try:
                    text_bbox = draw.textbbox((x0p, y0p), label, font=font)
                    tw = text_bbox[2] - text_bbox[0]
                    th = text_bbox[3] - text_bbox[1]
                except Exception:
                    tw = draw.textlength(label, font=font)
                    th = 14

                y_text = max(0, y0p - th)
                draw.rectangle([x0p, y_text, x0p + tw, y_text + th], fill=(255, 255, 255, 200))
                draw.text((x0p, y_text), label, fill="red", font=font)

        if k is not None:
            fname = f"rank{k:02d}_idx_{i:06d}_img_{idx:02d}.png"
        else:
            fname = f"idx_{i:06d}_img_{idx:02d}.png"
        out_path = os.path.join(save_root, fname)
        pil.save(out_path)

def normalize_rewards_to_dict(rewards):
    def _to_tensor_1d(x):
        if isinstance(x, torch.Tensor):
            t = x
        elif isinstance(x, np.ndarray):
            t = torch.from_numpy(x)
        else:
            t = torch.as_tensor(x)
        return t.float().view(-1)

    if isinstance(rewards, dict):
        out = {k: _to_tensor_1d(v) for k, v in rewards.items()}
        if "avg" not in out and "output_rewards" in out:
            out["avg"] = out["output_rewards"]
        return out

    return {"avg": _to_tensor_1d(rewards)}

class DummyFuture:
    def __init__(self, rewards_dict, metadata):
        self._rewards_dict = rewards_dict
        self._metadata = metadata
    def result(self):
        return self._rewards_dict, self._metadata

import numpy as np
from collections import deque
import torch

class PerPromptStatTracker:
    def __init__(self, global_std=False):
        self.global_std = global_std
        self.stats = {}
        self.history_prompts = set()

    def update(self, prompts, rewards, type='grpo'):
        prompts = np.array(prompts)
        rewards = np.array(rewards, dtype=np.float64)
        unique = np.unique(prompts)
        advantages = np.empty_like(rewards)*0.0
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = []
            self.stats[prompt].extend(prompt_rewards)
            self.history_prompts.add(hash(prompt))  # Add hash of prompt to history_prompts
        for prompt in unique:
            self.stats[prompt] = np.stack(self.stats[prompt])
            prompt_rewards = rewards[prompts == prompt]  # Fix: Recalculate prompt_rewards for each prompt
            mean = np.mean(self.stats[prompt], axis=0, keepdims=True)
            if self.global_std:
                std = np.std(rewards, axis=0, keepdims=True) + 1e-4  # Use global std of all rewards
            else:
                std = np.std(self.stats[prompt], axis=0, keepdims=True) + 1e-4
            if type=='grpo':
                advantages[prompts == prompt] = (prompt_rewards - mean) / std
            elif type=='rwr':
                # advantages[prompts == prompt] = (prompt_rewards - mean) / std
                advantages[prompts == prompt] = prompt_rewards
                # advantages[prompts == prompt] = torch.softmax(torch.tensor(prompt_rewards), dim=0).numpy()
            elif type=='sft':
                advantages[prompts == prompt] = (torch.tensor(prompt_rewards) == torch.max(torch.tensor(prompt_rewards))).float().numpy()
            elif type=='dpo':
                # Get the advantages of the current prompt
                prompt_advantages = torch.tensor(prompt_rewards)
                # Find the indices of the maximum and minimum values
                max_idx = torch.argmax(prompt_advantages)
                min_idx = torch.argmin(prompt_advantages)
                # If all rewards in a group are the same
                if max_idx == min_idx:
                    min_idx = 0
                    max_idx = 1
                result = torch.zeros_like(prompt_advantages).float()
                # Set the maximum index to 1, minimum index to -1
                result[max_idx] = 1.0
                result[min_idx] = -1.0
                advantages[prompts == prompt] = result.numpy()
                # print("reward difference one group", prompt_advantages[max_idx]-prompt_advantages[min_idx])
            
        return advantages

    def get_stats(self):
        avg_group_size = sum(len(v) for v in self.stats.values()) / len(self.stats) if self.stats else 0
        history_prompts = len(self.history_prompts)
        return avg_group_size, history_prompts
    
    def clear(self):
        self.stats = {}

def main():
    tracker = PerPromptStatTracker()
    prompts = ['a', 'b', 'a', 'c', 'b', 'a']
    rewards = [1, 2, 3, 4, 5, 6]
    advantages = tracker.update(prompts, rewards)
    print("Advantages:", advantages)
    avg_group_size, history_prompts = tracker.get_stats()
    print("Average Group Size:", avg_group_size)
    print("History Prompts:", history_prompts)
    tracker.clear()
    print("Stats after clear:", tracker.stats)

if __name__ == "__main__":
    main()