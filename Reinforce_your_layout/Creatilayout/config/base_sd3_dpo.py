import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    ############ General ############
    config.run_name = "" # run name for wandb logging and checkpoint saving
    config.seed = 42
    config.logdir = "logs" # top-level logging directory for checkpoint saving.
    config.save_freq = 1
    config.num_checkpoint_limit = 100000000 # number of checkpoints to keep before overwriting old ones.
    config.prompt_path = "path_to_filtered_data.json"

    ############ training config ############
    # number of epochs to train for. each epoch is one round of sampling from the model followed by training on those samples.
    config.num_epochs = 500
    config.mixed_precision = "bf16" # mixed precision training. options are "fp16", "bf16", and "no".
    config.allow_tf32 = True # allow tf32 on Ampere GPUs, which can speed up training.
    config.resume_from = "" # resume training from a checkpoint.
    config.use_lora = True
    config.use_xformers = False # whether or not to use xFormers to reduce memory usage.
    config.lora_rank = 32
    config.gradient_checkpointing = True # gradient checkpointing. this reduces memory usage at the cost of some additional compute.

    ############ Pretrained Model ############
    config.pretrained = pretrained = ml_collections.ConfigDict()
    pretrained.base_model = "stabilityai/stable-diffusion-3-medium-diffusers"
    pretrained.transformer = "HuiZhang0812/CreatiLayout"

    ############ Sampling ############
    config.sample = sample = ml_collections.ConfigDict()
    sample.num_steps = 40 # number of sampler inference steps.
    sample.guidance_scale = 4.5
    sample.s_churn = 0.7 # control the intensity of additional noise added during sampling
    sample.train_batch_size = 4
    sample.num_image_per_prompt = 2
    sample.num_batches_per_epoch = 1 # number of batches to sample per epoch. the total number of samples per epoch is `num_batches_per_epoch * batch_size * num_gpus`.
    sample.denoising_split = 1.0
    config.hmn_size = 2 # inference size
    config.hmn_pairs = 1 # pick Top hmn_pairs and Least hmn_pairs to do HMN
    sample.global_std = True
    sample.same_latent = False
    sample.sde_window_size = 2
    sample.sde_window_range = (0, 10)

    ############ Dataset and Reward ############
    config.dataset_name = "Nerako/dataset"
    config.dataset_split = "train"
    config.prompt_fn = "LayoutSAM"
    config.prompt_fn_kwargs = {}
    config.resolution = 1024
    config.reward_fn = "IoU_reward"

    ############ Training ############
    config.train = train = ml_collections.ConfigDict()

    ### <-Adam-> ###
    train.use_8bit_adam = False
    train.learning_rate = 3e-4
    train.adam_beta1 = 0.9
    train.adam_beta2 = 0.999
    train.adam_weight_decay = 1e-4
    train.adam_epsilon = 1e-8

    train.gradient_accumulation_steps = 1 # number of gradient accumulation steps. the effective batch size is `batch_size * num_gpus * gradient_accumulation_steps`.
    train.max_grad_norm = 1.0
    train.method = "dpo"

    ### <-train para-> ###
    train.dpo_beta = 10
    train.dpo_eps = 0.1
    train.dpso_lmbda = 1.0
    train.dpso_gamma = 5000
    train.ppo_adv_clip_max = 5
    train.ppo_clip_range = 3e-4

    train.num_inner_epochs = 1 # How many times should the same batch of sampled data be trained?
    train.cfg = True # whether or not to use classifier-free guidance during training
    train.save_interval = 50
    train.sample_path = ""
    train.json_path = ""
    train.adv_clip_max = 5
    train.clip_range = 1e-4
    train.lora_path = ""

    config.per_prompt_stat_tracking = True
    config.train.ref_update_step=10000000
    config.train.algorithm = 'dpo'
    config.train.batch_size = config.sample.train_batch_size
    config.train.timestep_fraction = 0.99
    config.train.beta = 100
    config.train.beta_dspo = 5000
    config.train.ema=True
    config.save_dir = "path_to_save_dir"

    return config
