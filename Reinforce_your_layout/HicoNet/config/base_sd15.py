import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    ############ General ############
    config.run_name = "" # run name for wandb logging and checkpoint saving
    config.seed = 42
    config.logdir = "logs" # top-level logging directory for checkpoint saving.
    config.save_freq = 1
    config.num_checkpoint_limit = 100000000 # number of checkpoints to keep before overwriting old ones.

    ############ training config ############
    # number of epochs to train for. each epoch is one round of sampling from the model followed by training on those samples.
    config.num_epochs = 200
    config.mixed_precision = "bf16" # mixed precision training. options are "fp16", "bf16", and "no".
    config.allow_tf32 = True # allow tf32 on Ampere GPUs, which can speed up training.
    config.resume_from = "" # resume training from a checkpoint.
    config.use_lora = True
    config.use_xformers = False # whether or not to use xFormers to reduce memory usage.
    config.lora_rank = 32
    config.gradient_checkpointing = True # gradient checkpointing. this reduces memory usage at the cost of some additional compute.

    ############ Pretrained Model ############
    config.pretrained = pretrained = ml_collections.ConfigDict()
    pretrained.controlnet = "qihoo360/HiCo_T2I"
    pretrained.base_model = "krnl/realisticVisionV51_v51VAE"

    ############ Sampling ############
    config.sample = sample = ml_collections.ConfigDict()
    sample.fuse_type = "avg"
    sample.mode = "single"
    sample.guess_mode = False
    sample.num_steps = 30 # number of sampler inference steps.
    sample.guidance_scale = 7.5
    sample.s_churn = 0.1 # control the intensity of additional noise added during sampling
    sample.batch_size = 2 # count 2 !!!!!!!!!!!!!!!
    sample.num_batches_per_epoch = 8 # number of batches to sample per epoch. the total number of samples per epoch is `num_batches_per_epoch * batch_size * num_gpus`.
    sample.denoising_split = 1.0
    config.hmn_size = 4 # inference size
    config.hmn_pairs = 1 # pick Top hmn_pairs and Least hmn_pairs to do HMN

    ############ Dataset and Reward ############
    config.dataset_name = "Nerako/dataset"
    config.dataset_split = "train"
    config.prompt_fn = "LayoutSAM"
    config.prompt_fn_kwargs = {}
    config.resolution = 512
    config.reward_fn = "IoU_reward"

    ############ Training ############
    config.train = train = ml_collections.ConfigDict()
    config.do_classifier_free_guidance = True

    ### <-Adam-> ###
    train.use_8bit_adam = False
    train.learning_rate = 1e-5
    train.adam_beta1 = 0.9
    train.adam_beta2 = 0.999
    train.adam_weight_decay = 1e-4
    train.adam_epsilon = 1e-8

    train.batch_size = 2  # batch size (per GPU!) to use for training.
    train.gradient_accumulation_steps = 8 # number of gradient accumulation steps. the effective batch size is `batch_size * num_gpus * gradient_accumulation_steps`.
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
    train.timestep_fraction = sample.denoising_split
    train.save_interval = 50
    train.sample_path = ""
    train.json_path = ""

    return config
