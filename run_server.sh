eval "$(path_to_conda_bin/conda shell.bash hook)"
conda activate creatilayout

export HF_HOME = path_to_hf_cache_dir
mkdir -p $HF_HOME

nohup python -m Server.server \
  --cuda-devices "0" \
  --dino-config path_to_groundingdino_config.py \
  --dino-ckpt path_to_groundingdino_ckpt.pth \
  --host 0.0.0.0 \
  --port 37777 \
  > groundingdino.log 2>&1 &
disown
