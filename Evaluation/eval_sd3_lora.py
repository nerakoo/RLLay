# run_infer_sd3_lora.py
# -*- coding: utf-8 -*-
import os, sys, json, argparse
from typing import Any, Dict, List, Tuple

import torch
from peft import PeftModel
from PIL import Image

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
for p in [
    "path_to_project_root",
    "path_to_creatilayout_root",
    PROJECT_ROOT,
]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.models.transformer_sd3_SiamLayout import SiamLayoutSD3Transformer2DModel
from src.pipeline.pipeline_sd3_CreatiLayout import CreatiLayoutSD3Pipeline

def example_to_prompt_and_layout(sample: Dict[str, Any], reso: int) -> Tuple[str, List[str], List[List[float]]]:
    meta = sample.get("metadata", {})
    img_info = meta.get("image_info", {})
    w = float(img_info.get("width", reso)) or float(reso)
    h = float(img_info.get("height", reso)) or float(reso)

    main_prompt = str(meta.get("global_caption", ""))

    caps, boxes = [], []
    for box in meta.get("bbox_info", []):
        x1, y1, x2, y2 = box.get("bbox", [0, 0, 0, 0])
        label = box.get("description") or box.get("label") or box.get("category") or "object"
        nx1, ny1 = float(x1) / w, float(y1) / h
        nx2, ny2 = float(x2) / w, float(y2) / h
        caps.append(str(label))
        boxes.append([nx1, ny1, nx2, ny2])
    return main_prompt, caps, boxes

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
def parse_args():
    p = argparse.ArgumentParser("SD3 + LoRA inference (no reward, no bbox drawing)")
    p.add_argument("--base_model", required=True, help="e.g. stabilityai/stable-diffusion-3-medium-diffusers")
    p.add_argument("--transformer", required=True, help="e.g. HuiZhang0812/CreatiLayout (has subfolder SiamLayout_SD3)")
    p.add_argument("--lora_dir", required=True, help="path to LoRA (adapter_name=learner)")
    p.add_argument("--prompts_json", required=True, help="path to prompts json (list of samples with 'metadata')")
    p.add_argument("--output_dir", default="infer_outputs_sd3")

    p.add_argument("--num_images", type=int, default=3, help="")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--guidance_scale", type=float, default=4.5)
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    p.add_argument("--max_prompts", type=int, default=-1, help="")
    p.add_argument("--reverse", action="store_true", help="")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device, dtype = choose_device_and_dtype(args.precision)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    # ====== Transformer + LoRA ======
    transformer = SiamLayoutSD3Transformer2DModel.from_pretrained(
        args.transformer,
        subfolder="SiamLayout_SD3",
        torch_dtype=dtype,
        attention_type="layout",
        strict=True,
    )
    transformer = PeftModel.from_pretrained(transformer, args.lora_dir, adapter_name="learner")
    transformer.set_adapter("learner")

    # ====== Pipeline ======
    pipe = CreatiLayoutSD3Pipeline.from_pretrained(
        args.base_model,
        transformer=transformer,
        torch_dtype=dtype,
    ).to(device)

    try:
        pipe.vae.to(device=device, dtype=dtype)
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
    except Exception:
        pass
    pipe.safety_checker = None

    # autocast
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        amp_ctx = lambda: torch.autocast(device_type="cuda", dtype=dtype)
    else:
        amp_ctx = torch.no_grad

    with open(args.prompts_json, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    assert isinstance(all_data, list) and len(all_data) > 0
    total = len(all_data) if args.max_prompts < 0 else min(args.max_prompts, len(all_data))

    N = len(all_data)
    if args.reverse:
        full_order = list(range(N, 0, -1))      # [N, N-1, ..., 1]
    else:
        full_order = list(range(1, N + 1))      # [1, 2, ..., N]
    index_list = full_order[:total]             
    index_list = [i for i in index_list if (os.getenv("ID_FROM") is None or i >= int(os.getenv("ID_FROM"))) and (os.getenv("ID_TO") is None or i <= int(os.getenv("ID_TO")))]

    for idx in index_list:
        it = idx - 1 
        sample = all_data[it]
        main_prompt, region_caps, region_boxes01 = example_to_prompt_and_layout(sample, reso=args.resolution)
        save_dir = os.path.join(args.output_dir, f"idx_{idx:06d}")  
        os.makedirs(save_dir, exist_ok=True)

        for k in range(args.num_images):
            seed_k = args.seed + it * 1000 + k
            gen = torch.Generator(device=device).manual_seed(seed_k)

            with torch.no_grad():
                with amp_ctx():
                    out = pipe(
                        prompt=[main_prompt],                
                        num_inference_steps=args.steps,
                        guidance_scale=args.guidance_scale,
                        height=args.resolution,
                        width=args.resolution,
                        bbox_phrases=region_caps,          
                        bbox_raw=region_boxes01,            
                        generator=gen,
                    )
            img = out.images[0]
            assert isinstance(img, Image.Image)

            img_path = os.path.join(save_dir, f"seed_{seed_k}.png")
            img.save(img_path)
            del out 

        print(f"[{idx}/{len(all_data)}] saved {args.num_images} images to: {save_dir}")

    print("======== DONE ========")

if __name__ == "__main__":
    main()
