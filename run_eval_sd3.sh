eval "$(path_to_conda_bin/conda shell.bash hook)"

#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0
export SERVICE_HOST=localhost
export SERVICE_PORT=37777

export HF_HOME=path_to_hf_cache_dir
export HF_TOKEN="path_to_hf_token"
mkdir -p $HF_HOME

ID_FROM=7900 ID_TO=8480 \
python Evaluation/run_infer_sd3_lora.py \
  --base_model stabilityai/stable-diffusion-3-medium-diffusers \
  --transformer HuiZhang0812/CreatiLayout \
  --lora_dir path_to_lora_dir \
  --prompts_json path_to_filtered_data.json \
  --output_dir path_to_output_dir \
  --num_images 3 --steps 40 --guidance_scale 4.5 --resolution 1024 \
  --seed 0 --precision bf16 \
