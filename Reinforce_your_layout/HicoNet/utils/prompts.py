from importlib import resources
import os
import functools
import random
import inflect
import json

def LayoutSAM(
    json_file="path_to_filtered_data.json",
    final_w=1024,
    final_h=1024
):
    with open(json_file, 'r', encoding='utf-8') as f:
        datas = json.load(f)

    random_data = random.choice(datas)

    meta_info = random_data.get("metadata", {}).get("image_info", {})
    old_width = meta_info.get("width", 0)
    old_height = meta_info.get("height", 0)

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
    metadata = random_data

    return main_prompt, metadata