import os
import base64
from io import BytesIO
from PIL import Image
import torch
import numpy as np
import requests
from typing import Union, Dict

HOST = os.getenv("SERVICE_HOST", "localhost")
PORT = os.getenv("SERVICE_PORT", "37777")
REWARD_ENDPOINT = f"http://{HOST}:{PORT}/reward"

def to_pil(img_tensor: torch.Tensor) -> Image.Image:
    arr = img_tensor.detach().cpu().to(torch.float32).numpy()
    arr = arr.transpose(1, 2, 0)
    if arr.max() <= 1.0:
        arr = (arr * 255).round()
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def pil_to_base64(img: Image.Image, fmt: str = "JPEG") -> str:
    buf = BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def call_reward(
    image: Union[str, Image.Image],
    meta: Dict[str, any],
    connect_timeout: float = 5.0,
    read_timeout: float = 300.0
) -> float:
    if isinstance(image, Image.Image):
        image = pil_to_base64(image)

    if isinstance(image, str) and os.path.exists(image):
        payload = {"image_path": image, "meta": meta}
    else:
        payload = {"image_b64": image, "meta": meta}

    resp = requests.post(
        REWARD_ENDPOINT,
        json=payload,
        timeout=(connect_timeout, read_timeout)
    )
    resp.raise_for_status()
    data = resp.json()
    if "reward" not in data:
        raise RuntimeError(f"Unexpected response: {data}")
    return data["reward"]

def IoU_reward(device: str = "cuda"):
    def _fn(images, prompts, prompt_metadata):
        batch_size = len(images)
        scores = []

        for i in range(batch_size):
            img  = images[i]
            meta = prompt_metadata[i]

            if isinstance(img, torch.Tensor):
                img = to_pil(img)

            score_i = call_reward(img, meta)

            if not isinstance(score_i, torch.Tensor):
                score_i = torch.tensor(score_i, device=device)
            scores.append(score_i)

        scores_tensor = torch.stack(scores, dim=0)
        print("paired rewards:", scores_tensor)
        return scores_tensor, {}
    return _fn
