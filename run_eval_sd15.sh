eval "$(path_to_conda_bin/conda shell.bash hook)"

#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=2
export SERVICE_HOST=localhost
export SERVICE_PORT=37777

export HF_HOME=path_to_hf_cache_dir
export HF_TOKEN="path_to_hf_token"
mkdir -p $HF_HOME

BASE_MODEL="krnl/realisticVisionV51_v51VAE"
CONTROLNET="qihoo360/HiCo_T2I"
LORA_PATH="path_to_lora_checkpoint"
PROMPTS_JSON="path_to_filtered_data.json"      
OUT_DIR="eval_outputs"

NUM_ITERS=50      
NUM_IMAGES=3     
STEPS=30         
GUIDANCE=7.5     
RESO=512
SEED=42
PRECISION="bf16" 

python -u Evaluation/eval_sd15_lora.py \
  --base_model "krnl/realisticVisionV51_v51VAE" \
  --controlnet "qihoo360/HiCo_T2I" \
  --lora_path "path_to_lora_checkpoint" \
  --prompts_json "path_to_filtered_data.json" \
  --output_dir "path_to_eval_output_dir" \
  --steps 30 \
  --guidance_scale 7.5 \
  --resolution 512 \
  --precision bf16 \
  --num_images 3 \
  --max_prompts -1 \
  --send_image_path