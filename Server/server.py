import os
import argparse
from typing import Any, Dict, Optional, Union
from io import BytesIO
from threading import Lock
import base64

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import uvicorn
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description="Launch LayoutDPO Reward Service")
    parser.add_argument(
        "--cuda-devices",
        type=str,
        default=os.getenv("CUDA_VISIBLE_DEVICES", ""),
        help="Comma-separated list of GPU ids to expose, e.g. '0,1'"
    )
    parser.add_argument("--dino-config", type=str, default=os.getenv("DINO_MODEL_CONFIG", ""))
    parser.add_argument("--dino-ckpt",   type=str, default=os.getenv("DINO_MODEL_CKPT", ""))
    parser.add_argument("--host",        type=str, default="0.0.0.0")
    parser.add_argument("--port",        type=int, default=37733)
    return parser.parse_args()

def create_app(dino_config: str, dino_ckpt: str, cuda_devices: Optional[str] = None) -> FastAPI:
    if cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
        print(f"🔧 CUDA_VISIBLE_DEVICES set to: {cuda_devices}")

    from Server.reward_models.GroundingDINO.groundingDINO import GroundingDINOEvaluator
    import torch

    print("🧪 torch sees", torch.cuda.device_count(), "visible GPU(s)")
    if torch.cuda.is_available():
        print("🟢 torch.cuda.current_device =", torch.cuda.current_device())

    print("⏳ Loading GroundingDINO model...")
    evaluator = GroundingDINOEvaluator(
        model_config_path=dino_config,
        model_ckpt_path=dino_ckpt,
    )
    print("✅ GroundingDINO model loaded.")

    reward_lock = Lock()
    app = FastAPI(title="LayoutDPO Reward Service")

    class RewardReq(BaseModel):
        image_path: Optional[str] = None
        image_b64:  Optional[str] = None
        meta:       Dict[str, Any]

    class RewardResp(BaseModel):
        reward: float

    @app.post("/ping")
    def ping():
        return {"ok": 1}

    @app.post("/reward", response_model=RewardResp)
    def reward_endpoint(req: RewardReq):
        if req.image_b64:
            try:
                img = Image.open(BytesIO(base64.b64decode(req.image_b64))).convert("RGB")
                image_input = img
            except Exception as e:
                raise HTTPException(400, f"Invalid base64 image: {e}")
        elif req.image_path:
            image_input = req.image_path
        else:
            raise HTTPException(400, "Must provide image_path or image_b64")

        with reward_lock:
            try:
                score = LayoutDPO_reward(image_input, req.meta, evaluator)
            except Exception as e:
                raise HTTPException(500, f"Error computing reward: {e}")

        return {"reward": score}

    return app

from typing import Tuple
def LayoutDPO_reward(
    image_input: Union[str, Image.Image],
    meta: Dict[str, Any],
    evaluator
) -> float:
    image_source, _ = evaluator.load_input_image(image_input)
    img_h, img_w = image_source.shape[:2]
    results = evaluator.evaluate_image(image_source, meta)

    total_score, valid_count = 0.0, 0
    for item in results:
        if item.get("gt_bbox") is None or item.get("iou") is None:
            continue
        x1, y1, x2, y2 = map(int, item["gt_bbox"])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image_source[y1:y2, x1:x2]
        _ = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)) 
        prob = 1.0
        iou_val = float(item["iou"].item()) if hasattr(item["iou"], "item") else float(item["iou"])
        total_score += iou_val * prob
        valid_count += 1

    return 0.0 if valid_count == 0 else total_score / valid_count

if __name__ == "__main__":
    args = parse_args()
    print("🛠️ Arguments parsed:")
    print(f"   CUDA devices: {args.cuda_devices}")
    print(f"   DINO config:  {args.dino_config}")
    print(f"   DINO ckpt:    {args.dino_ckpt}")
    print(f"   Host:         {args.host}")
    print(f"   Port:         {args.port}")

    app = create_app(
        dino_config=args.dino_config,
        dino_ckpt=args.dino_ckpt,
        cuda_devices=args.cuda_devices,
    )
    print(f"🚀 Starting service on {args.host}:{args.port}...")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
