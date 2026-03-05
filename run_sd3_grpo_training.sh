eval "$(path_to_conda_bin/conda shell.bash hook)"
conda activate path_to_env_name

#!/usr/bin/env bash
set -e

export HF_HOME=path_to_hf_cache_dir
export HF_TOKEN="path_to_hf_token"
mkdir -p $HF_HOME

export PYTHONPATH="path_to_creatilayout_root:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="0,2"

CONFIG_PATH="path_to_base_sd3_grpo_config.py"

accelerate launch \
  --num_processes 2 \
  --main_process_port 33777 \
  Reinforce_your_layout/Creatilayout/GRPO_training.py \
  --config="$CONFIG_PATH"
