eval "$(path_to_conda_bin/conda shell.bash hook)"
conda activate path_to_env_name

#!/usr/bin/env bash
set -e

export HF_HOME=path_to_hf_cache_dir
export HF_TOKEN="path_to_hf_token"
mkdir -p $HF_HOME

export PYTHONPATH="/mnt/pentagon/nerako/Layout_project/Reinforce_your_layout/HicoNet:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

export PYTHONPATH="path_to_hiconet_root:$PYTHONPATH"

accelerate launch \
  --num_processes 8 \
  --main_process_port 29999 \
  Reinforce_your_layout/HicoNet/ARPO_training.py \
  --config="$CONFIG_PATH"
