import torch
import os
from CreatiLayout.utils.bbox_visualization import bbox_visualization, scale_boxes
from PIL import Image
from CreatiLayout.src.models.transformer_sd3_SiamLayout import SiamLayoutSD3Transformer2DModel
from CreatiLayout.src.pipeline.pipeline_sd3_CreatiLayout import CreatiLayoutSD3Pipeline

import random
import json
from jinja2.compiler import generate

def pick_random_and_resize(
    json_file = "path_to_json",
    final_w=512,
    final_h=512
):
    with open(json_file, 'r', encoding='utf-8') as f:
        datas = json.load(f)

    print(len(datas))

    for i in range(10):
        print(datas[i])

    random_data = random.choice(datas)

    meta_info = random_data.get("metadata", {}).get("image_info", {})
    old_width = meta_info.get("width", 0)
    old_height = meta_info.get("height", 0)

    if old_width == 0 or old_height == 0:
        return None, None

    scale_x = final_w / old_width
    scale_y = final_h / old_height

    bbox_info_list = random_data.get("metadata", {}).get("bbox_info", [])
    for box_info in bbox_info_list:
        if "bbox" in box_info and len(box_info["bbox"]) == 4:
            x1, y1, x2, y2 = box_info["bbox"]
            new_x1 = round(x1 * scale_x, 2)
            new_y1 = round(y1 * scale_y, 2)
            new_x2 = round(x2 * scale_x, 2)
            new_y2 = round(y2 * scale_y, 2)
            box_info["bbox"] = [new_x1, new_y1, new_x2, new_y2]

    meta_info["width"] = final_w
    meta_info["height"] = final_h

    main_prompt = random_data.get("metadata", {}).get("global_caption", "")
    # metadata = random_data.get("metadata", {})
    metadata = random_data

    return main_prompt, metadata

def generate_data():
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

    model_path = "stabilityai/stable-diffusion-3-medium-diffusers"
    ckpt_path = "HuiZhang0812/CreatiLayout"
    transformer_additional_kwargs = dict(attention_type="layout", strict=True)
    transformer = SiamLayoutSD3Transformer2DModel.from_pretrained(
        ckpt_path, subfolder="SiamLayout_SD3", torch_dtype=torch.float16, **transformer_additional_kwargs)
    pipe = CreatiLayoutSD3Pipeline.from_pretrained(model_path, transformer=transformer, torch_dtype=torch.float16)
    pipe = pipe.to(device)

if __name__ == "__main__":
    pick_random_and_resize()