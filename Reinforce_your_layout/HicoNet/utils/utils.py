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

def get_meta(global_caption, region_caption_list, region_bboxes_list, resolution):
    if len(region_caption_list) != len(region_bboxes_list):
        raise ValueError("region_caption_list 和 region_bboxes_list 的长度必须一致。")

    meta = {}
    meta["global_caption"] = global_caption
    meta["image_info"] = {"width": resolution, "height": resolution}

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