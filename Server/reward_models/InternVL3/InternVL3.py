import os
import re
from typing import Union

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# Local cache directory for model weights (replace with your own path)
_CACHE_DIR = "path_to_internvl_cache_dir"   
_CKPT = "OpenGVLab/InternVL3-38B-hf"

class InternVL3Matcher():
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        print("⏳ Loading InternVL-3-38B-hf …")
        self.processor = AutoProcessor.from_pretrained(
            _CKPT, cache_dir=_CACHE_DIR, trust_remote_code=True
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            _CKPT,
            cache_dir=_CACHE_DIR,
            device_map="auto",
            max_memory={0: "45GiB", 1: "45GiB"},
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval()
        print("✅ InternVL-3-38B-hf ready.")

    @torch.inference_mode()
    def match_score(
        self,
        image: Union[str, Image.Image],
        description: str,
        max_new_tokens: int = 6,
    ) -> float:
        if isinstance(image, str):
            image = Image.open(image)

        prompt = (
            f"Please assess how well the given image matches the following description:\n"
            f"Description: {description}\n\n"
            f"Provide an integer score between 0 and 10 (0 = completely mismatched, 10 = perfectly matched).\n"
            f"Respond with only the number, without any additional text."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        chat_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        inputs = self.processor(
            text=[chat_text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        inputs = {
            k: (v.to(self.model.device, dtype=torch.bfloat16)
                if v.dtype.is_floating_point else v.to(self.model.device))
            for k, v in inputs.items()
        }

        gen_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )
        answer = self.processor.decode(
            gen_ids[0, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        match = re.search(r"0?\.\d+|1(?:\.0+)?", answer)
        score = float(match.group()) if match else 0.0
        return max(0.0, min(score, 10.0))

# if __name__ == "__main__":
#     matcher = InternVL3Matcher()
#     img = "/mnt/pentagon/nerako/DDPO_baseline/SD3_based/benchmark/models/qwen/test.jpg"
#     desc = "a cat"
#
#     for _ in range(5):
#         s = matcher.match_score(img, desc)
#         print(f"match: {s:.3f}")
