import os
import json
import copy
from typing import Any, Dict, List, Tuple, Optional

from torch.utils.data import Dataset

class LayoutJSONPromptDataset(Dataset):
    def __init__(self, json_path: str, final_w: int = 1024, final_h: int = 1024):
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list) or len(data) == 0:
            raise ValueError("JSON must be a non-empty list of samples.")

        self.data: List[Dict[str, Any]] = data
        self.final_w = int(final_w)
        self.final_h = int(final_h)

    def __len__(self) -> int:
        return len(self.data)

    def _scale_one_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        item = copy.deepcopy(sample)
        meta = item.setdefault("metadata", {})
        img_info = meta.setdefault("image_info", {})

        old_w = float(img_info.get("width", 0) or 0)
        old_h = float(img_info.get("height", 0) or 0)

        if old_w <= 0:
            old_w = float(self.final_w)
        if old_h <= 0:
            old_h = float(self.final_h)

        sx = self.final_w / old_w
        sy = self.final_h / old_h

        bbox_list = meta.get("bbox_info", [])
        if isinstance(bbox_list, list):
            for box in bbox_list:
                bb = box.get("bbox")
                if isinstance(bb, list) and len(bb) == 4:
                    x1, y1, x2, y2 = bb
                    box["bbox"] = [
                        round(float(x1) * sx, 2),
                        round(float(y1) * sy, 2),
                        round(float(x2) * sx, 2),
                        round(float(y2) * sy, 2),
                    ]

        img_info["width"] = self.final_w
        img_info["height"] = self.final_h
        return item

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raw = self.data[idx]
        scaled = self._scale_one_sample(raw)

        meta = scaled.get("metadata", {})
        prompt = meta.get("global_caption", "")
        
        if not isinstance(prompt, str) or prompt == "":
            prompt = scaled.get("prompt", "")

        return {"prompt": str(prompt), "metadata": scaled}

    @staticmethod
    def collate_fn(examples: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
        prompts = [ex["prompt"] for ex in examples]
        metadatas = [ex["metadata"] for ex in examples]
        return prompts, metadatas
