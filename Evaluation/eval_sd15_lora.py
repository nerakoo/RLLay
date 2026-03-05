# -*- coding: utf-8 -*-
import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))  
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import argparse
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
import numpy as np
import base64
import requests
from io import BytesIO

from diffusers import DPMSolverMultistepScheduler
from diffusers.utils import convert_unet_state_dict_to_peft
from peft import set_peft_model_state_dict
from Reinforce_your_layout.HicoNet.utils.utils import Hicoinput, get_meta
from Reinforce_your_layout.HicoNet.src.HiCo_T2I.diffusers.src.diffusers.pipelines.controlnet.pipeline_hiconet_layout import (
    ControlNetModel,
    StableDiffusionHicoNetLayoutPipeline,
)

# ----------------------------
# Reward（HTTP）
# ----------------------------
HOST = os.getenv("SERVICE_HOST", "localhost")
PORT = os.getenv("SERVICE_PORT", "37777")
REWARD_ENDPOINT = f"http://{HOST}:{PORT}/reward"

def pil_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = BytesIO()
    img.save(buf, format=fmt)  
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def call_reward_http(
    image_or_path: Any,
    meta: Dict[str, Any],
    use_image_path: bool,
    connect_timeout: float = 5.0,
    read_timeout: float = 300.0,
) -> float:
    if use_image_path and isinstance(image_or_path, str) and os.path.exists(image_or_path):
        payload = {"image_path": image_or_path, "meta": meta}
    else:
        if isinstance(image_or_path, Image.Image):
            img_b64 = pil_to_base64(image_or_path)
        else:
            with Image.open(image_or_path) as im:
                img_b64 = pil_to_base64(im)
        payload = {"image_b64": img_b64, "meta": meta}

    r = requests.post(REWARD_ENDPOINT, json=payload, timeout=(connect_timeout, read_timeout))
    r.raise_for_status()
    data = r.json()
    if "reward" not in data:
        raise RuntimeError(f"Unexpected response: {data}")
    return float(data["reward"])

def IoU_reward_http(device: torch.device, use_image_path: bool):
    def _fn(images_or_paths: List[Any], prompts: List[str], metas: List[Dict[str, Any]]):
        scores = []
        for img_or_path, meta in zip(images_or_paths, metas):
            score = call_reward_http(img_or_path, meta, use_image_path=use_image_path)
            scores.append(torch.tensor(score, device=device, dtype=torch.float32))
        scores_t = torch.stack(scores, dim=0)
        print("paired rewards:", scores_t)
        return scores_t, {}
    return _fn

def example_to_prompt_and_layout(sample: Dict[str, Any], reso: int) -> Tuple[str, List[str], List[List[float]]]:
    meta = sample.get("metadata", {})
    img_info = meta.get("image_info", {})
    w = float(img_info.get("width", reso))
    h = float(img_info.get("height", reso))
    if w <= 0: w = float(reso)
    if h <= 0: h = float(reso)
    main_prompt = str(meta.get("global_caption", ""))

    caps, boxes = [], []
    for box in meta.get("bbox_info", []):
        x1, y1, x2, y2 = box.get("bbox", [0, 0, 0, 0])
        label = box.get("text") or box.get("label") or box.get("category") or "object"
        nx1, ny1 = float(x1) / w, float(y1) / h
        nx2, ny2 = float(x2) / w, float(y2) / h
        caps.append(str(label))
        boxes.append([nx1, ny1, nx2, ny2])
    return main_prompt, caps, boxes

def try_load_lora(pipeline, lora_path: str):
    try:
        pipeline.load_lora_weights(lora_path)
        print(f"[LoRA] Loaded via pipeline.load_lora_weights from {lora_path}")
        return
    except Exception as e:
        print(f"[LoRA] pipeline.load_lora_weights failed: {e}. Trying fallback...")

    lora_pack = StableDiffusionHicoNetLayoutPipeline.lora_state_dict(lora_path)
    if isinstance(lora_pack, tuple):
        lora_state_dict = lora_pack[0]
    else:
        lora_state_dict = lora_pack

    if "unet" in lora_state_dict and isinstance(lora_state_dict["unet"], dict):
        unet_state = lora_state_dict["unet"]
    else:
        unet_state = {k.replace("unet.", "", 1): v for k, v in lora_state_dict.items() if k.startswith("unet.")}

    unet_state = convert_unet_state_dict_to_peft(unet_state)
    set_peft_model_state_dict(pipeline.unet, unet_state, adapter_name="default")
    print(f"[LoRA] Loaded via fallback from {lora_path}")

def choose_device_and_dtype(precision: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        if precision == "bf16" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif precision == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.float32
    else:
        dtype = torch.float32
    return device, dtype

# ----------------------------
# Argparse
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser("Simple LoRA Evaluation on single GPU")
    p.add_argument("--base_model", required=True)
    p.add_argument("--controlnet", required=True)
    p.add_argument("--lora_path", required=True)
    p.add_argument("--prompts_json", required=True)
    p.add_argument("--output_dir", default="eval_outputs")

    p.add_argument("--max_prompts", type=int, default=-1, help="")
    p.add_argument("--num_images", type=int, default=3, help="")

    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    p.add_argument("--fuse_type", default="avg")
    p.add_argument("--infer_mode", default="single")
    p.add_argument("--send_image_path", action="store_true", help="")
    return p.parse_args()

# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device, dtype = choose_device_and_dtype(args.precision)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    # Pipeline
    controlnet = ControlNetModel.from_pretrained(
        args.controlnet,
        torch_dtype=dtype if device.type == "cuda" else torch.float32,
    )
    pipe = StableDiffusionHicoNetLayoutPipeline.from_pretrained(
        args.base_model,
        controlnet=[controlnet],
        torch_dtype=dtype if device.type == "cuda" else torch.float32,
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None
    try:
        pipe.enable_attention_slicing("max")
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
    except Exception:
        pass

    # LoRA
    try_load_lora(pipe, args.lora_path)
    reward_fn = IoU_reward_http(device, use_image_path=args.send_image_path)
    results_path = os.path.join(args.output_dir, "scores.jsonl")
    with open(results_path, "w", encoding="utf-8") as _f:
        pass
    total_sum = 0.0
    total_cnt = 0

    # autocast
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        amp_ctx = lambda: torch.autocast(device_type="cuda", dtype=dtype)
    else:
        amp_ctx = torch.no_grad

    with open(args.prompts_json, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    assert isinstance(all_data, list) and len(all_data) > 0

    total = len(all_data) if args.max_prompts < 0 else min(args.max_prompts, len(all_data))

    for it in range(total):
        sample = all_data[it]
        main_prompt, region_caps, region_boxes = example_to_prompt_and_layout(sample, reso=args.resolution)
        global_caption_list = [main_prompt]
        prompt, layo_prompt, list_cond_image_pil = Hicoinput(
            global_caption_list, region_caps, region_boxes, img_size=args.resolution
        )

        meta = get_meta(main_prompt, region_caps, region_boxes, args.resolution)
        images_or_paths: List[Any] = []
        seed_list: List[int] = []
        path_list: List[str] = []

        save_dir = os.path.join(args.output_dir, f"idx_{it+1:06d}")
        os.makedirs(save_dir, exist_ok=True)

        for k in range(args.num_images):
            seed_k = args.seed + it * 1000 + k
            seed_list.append(seed_k)
            gen = torch.Generator(device=device).manual_seed(seed_k)
            with torch.no_grad():
                with amp_ctx():
                    out = pipe(
                        prompt=prompt,
                        layo_prompt=layo_prompt,
                        guidance_scale=args.guidance_scale,
                        infer_mode=args.infer_mode,
                        num_inference_steps=args.steps,
                        image=list_cond_image_pil,
                        fuse_type=args.fuse_type,
                        width=args.resolution,
                        height=args.resolution,
                        generator=gen,
                    )
            img = out.images[0]
            assert isinstance(img, Image.Image)

            img_path = os.path.join(save_dir, f"seed_{seed_k}.png")
            img.save(img_path)
            path_list.append(img_path)
            images_or_paths.append(img_path if args.send_image_path else img)
            del out

        prompts_dup = [main_prompt] * args.num_images
        metas_dup = [meta] * args.num_images
        scores, _ = reward_fn(images_or_paths, prompts_dup, metas_dup)
        scores_list = [float(s) for s in scores.tolist()]

        with open(results_path, "a", encoding="utf-8") as fout:
            for k, sc in enumerate(scores_list):
                rec = {
                    "prompt_idx": it + 1,
                    "image_idx": k,
                    "seed": seed_list[k],
                    "reward": sc,
                    "image_path": path_list[k],
                    "prompt": main_prompt,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_sum += sc
                total_cnt += 1

        prompt_mean = float(np.mean(scores_list)) if scores_list else 0.0
        running_mean = (total_sum / total_cnt) if total_cnt > 0 else 0.0

        print(
            f"[{it+1}/{total}] "
            f"prompt_mean={prompt_mean:.4f} | global_mean={running_mean:.4f} | per-image={scores_list}"
        )

    overall = (total_sum / total_cnt) if total_cnt > 0 else 0.0
    print(f"======== DONE ======== overall_mean_reward={overall:.4f} over {total} prompts, {total_cnt} images")

    summary = {
        "overall_mean_reward": overall,
        "num_prompts": total,
        "num_images_per_prompt": args.num_images,
        "num_images_total": total_cnt,
        "seed": args.seed,
        "precision": args.precision,
        "send_image_path": args.send_image_path,
        "scores_jsonl": results_path,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
