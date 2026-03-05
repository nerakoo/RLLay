import os
import copy
import tempfile
import contextlib
import datetime
from collections import defaultdict
from functools import partial

import torch
import wandb
from PIL import Image
from absl import app, flags
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_unet_state_dict_to_peft
from diffusers.utils.torch_utils import is_compiled_module
from ml_collections import config_flags
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from tqdm import tqdm

import utils.prompts
import utils.rewards
from utils.store_util import CPUStore
from utils.utils import transform_meta, get_meta, annotate_and_save, Hicoinput
from training_patch.pipeline.sd15_pipeline_with_logprob import pipeline_with_logprob
from training_patch.logprob.dpm_step_with_logprob import dpm_step_with_logprob
from src.HiCo_T2I.diffusers.src.diffusers.pipelines.controlnet.pipeline_hiconet_layout import ControlNetModel, StableDiffusionHicoNetLayoutPipeline
from diffusers import DPMSolverMultistepScheduler

tqdm = partial(tqdm, dynamic_ncols=True)
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base_sd15.py", "Training configuration.")
logger = get_logger(__name__)

def main(_):
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    if config.resume_from:
        config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
        if "checkpoint_" not in os.path.basename(config.resume_from):
            checkpoints = list(
                filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from))
            )
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {config.resume_from}")
            config.resume_from = os.path.join(
                config.resume_from,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )

    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)
    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=False,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps
                                    * num_train_timesteps,
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="fine-tune-sd15-oft",
            config=config.to_dict(),
            init_kwargs={"wandb": {"name": config.run_name}},
        )
    logger.info(f"\n{config}")
    set_seed(config.seed, device_specific=True)

    HiCoNet = ControlNetModel.from_pretrained(config.pretrained.controlnet, torch_dtype=torch.bfloat16)
    pipeline = StableDiffusionHicoNetLayoutPipeline.from_pretrained(
        config.pretrained.base_model, controlnet=[HiCoNet], torch_dtype=torch.bfloat16
    ).to(accelerator.device)

    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(not config.use_lora)
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    if not config.resume_from:
        unet_ref = copy.deepcopy(pipeline.unet)
        for param in unet_ref.parameters():
            param.requires_grad = False
        unet_ref.to(accelerator.device, dtype=next(pipeline.unet.parameters()).dtype)
    unet = pipeline.unet

    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if config.use_lora:
        unet_lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"]
        )
        unet.add_adapter(unet_lora_config)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            unet_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(unwrap_model(unet))):
                    unet_lora_layers_to_save = get_peft_model_state_dict(model)
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")
                weights.pop()

            StableDiffusionHicoNetLayoutPipeline.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        def _unwrap(m):
            m = accelerator.unwrap_model(m)
            return m._orig_mod if is_compiled_module(m) else m

        unet_ = None
        while len(models) > 0:
            model = models.pop()
            if isinstance(model, type(_unwrap(unet))):
                unet_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")
        if unet_ is None:
            raise RuntimeError("UNet not found in models list for load_model_hook")

        lora_pack = StableDiffusionHicoNetLayoutPipeline.lora_state_dict(input_dir)
        if isinstance(lora_pack, tuple):
            lora_state_dict = lora_pack[0]
        else:
            lora_state_dict = lora_pack

        if "unet" in lora_state_dict and isinstance(lora_state_dict["unet"], dict):
            unet_state_dict = lora_state_dict["unet"]
        else:
            unet_state_dict = {k.replace("unet.", "", 1): v
                            for k, v in lora_state_dict.items()
                            if k.startswith("unet.")}

        try:
            unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        except Exception as e:
            print(f"[load_model_hook] convert_unet_state_dict_to_peft skipped: {e}")

        incompatible_keys = set_peft_model_state_dict(unet_, unet_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    "Loading adapter weights led to unexpected keys not found in the model: "
                    f"{unexpected_keys}"
                )

        if config.mixed_precision == "fp16":
            cast_training_params([unet_])

    # def load_model_hook(models, input_dir):
    #     unet_ = None
    #     text_encoder_one_ = None

    #     while len(models) > 0:
    #         model = models.pop()

    #         if isinstance(model, type(unwrap_model(unet))):
    #             unet_ = model
    #         else:
    #             raise ValueError(f"unexpected save model: {model.__class__}")

    #     lora_state_dict = StableDiffusionHicoNetLayoutPipeline.lora_state_dict(input_dir)

    #     unet_state_dict = {
    #         f'{k.replace("unet.", "")}': v for k, v in lora_state_dict.items() if k.startswith("unet.")
    #     }
    #     unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
    #     incompatible_keys = set_peft_model_state_dict(unet_, unet_state_dict, adapter_name="default")
    #     if incompatible_keys is not None:
    #         unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
    #         if unexpected_keys:
    #             logger.warning(
    #                 f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
    #                 f" {unexpected_keys}. "
    #             )

    #     if config.mixed_precision == "fp16":
    #         models = [unet_]
    #         cast_training_params(models)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Make sure the trainable params are in float32.
    if accelerator.mixed_precision == "fp16":
        models = [unet]
        cast_training_params(models, dtype=torch.float32)

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))
    optimizer = optimizer_cls(
        params_to_optimize,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # prepare prompt and reward fn
    prompt_fn = getattr(utils.prompts, config.prompt_fn)
    reward_fn = getattr(utils.rewards, config.reward_fn)(
        device=accelerator.device)  # ensure reward model is on same device

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    # autocast = accelerator.autocast

    # Prepare everything with our `accelerator`.
    unet = accelerator.prepare(unet)
    # Train!
    samples_per_epoch = (
            config.sample.batch_size
            * accelerator.num_processes
            * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
            config.train.batch_size
            * accelerator.num_processes
            * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )

    # assert config.sample.batch_size == config.train.batch_size 
    # assert config.sample.batch_size == 2  # for dpo loss
    # assert config.hmn_size >= config.hmn_pairs  # for HNM
    # assert config.train.gradient_accumulation_steps == config.hmn_pairs * config.hmn_pairs * config.sample.num_batches_per_epoch

    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        accelerator.load_state(config.resume_from)
        first_epoch = int(config.resume_from.split("_")[-1]) + 1
        unet_ref = copy.deepcopy(pipeline.unet)
        for param in unet_ref.parameters():
            param.requires_grad = False
        optimizer = optimizer_cls(
            params_to_optimize,
            lr=config.train.learning_rate,
            betas=(config.train.adam_beta1, config.train.adam_beta2),
            weight_decay=config.train.adam_weight_decay,
            eps=config.train.adam_epsilon,
        )
        optimizer = accelerator.prepare(optimizer)
    else:
        first_epoch = 0

    #######################test######################################
    # global_caption = [
    #     'This is a photo showcasing a mosque with a typical Islamic architectural style. The mosque is characterized by its towering minaret and large dome, both of which are made of stone. The entrance of the mosque is a double-arched porch, with a row of decorative pillars on both sides. The exterior walls of the building are made of stone, with a grey tone, and the roof is covered with a grey dome. The surrounding environment is a spacious square, with a few trees and some low-rise buildings visible in the background.'
    # ]
    # region_caption_list = ['trees', 'trees', 'minaret', 'large dome grey dome', 'spacious square']
    # region_bboxes_list = [
    #     [0.769603438013337, 0.4799966227213542, 0.9985172047334558, 0.920195556640625],
    #     [0.0008651763387257525, 0.603338134765625, 0.19105535514363658, 0.9425865071614583],
    #     [0.675440323964526, 0.06483316548665365, 0.7717202655663329, 0.6422011311848959],
    #     [0.3071458074118527, 0.48534086100260415, 0.6709005000146667, 0.6512798258463541],
    #     [0.00197142355134699, 0.9511194661458333, 0.9976715737200798, 0.9983336588541667]
    # ]

    # prompt, layo_prompt, list_cond_image_pil = Hicoinput(
    #     global_caption, region_caption_list, region_bboxes_list, config.resolution
    # )
    # print("list_cond_image_pil", list_cond_image_pil)

    # save_dir = "path_to_test_output_dir"
    # os.makedirs(save_dir, exist_ok=True)

    # images = []
    # with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    #     for k in range(config.hmn_size): 
    #         out = pipeline(
    #             prompt=prompt, layo_prompt=layo_prompt, guidance_scale=config.sample.guidance_scale, infer_mode="single", num_inference_steps=50,
    #             image=list_cond_image_pil, fuse_type=config.sample.fuse_type, width=512, height=512,
    #         )
    #         img = out.images[0]
    #         assert isinstance(img, Image.Image), f"Expected PIL.Image, got {type(img)}"
    #         images.append(img)

    #         out_path = os.path.join(save_dir, f"output_{k}.png")
    #         img.save(out_path)
    #         print("Saved to:", out_path)

    #         del out
    #         torch.cuda.empty_cache()

    # pil_img = images[0]
    # main_path = os.path.join(save_dir, "output.png")
    # pil_img.save(main_path)
    # print("Saved to:", main_path, "Exists:", os.path.exists(main_path))
    ################################################################

    global_step = 0
    for epoch in range(first_epoch, config.num_epochs):
        with CPUStore(keep_images=False, cast_dtype=torch.float16) as store:
            #################### SAMPLING ####################
            pipeline.unet.eval()
            samples = []
            samples_minibatch = []

            for i in tqdm(
                    range(config.sample.num_batches_per_epoch),
                    desc=f"Epoch {epoch}: sampling",
                    disable=not accelerator.is_local_main_process,
                    position=0,
            ):
                single_prompts, prompt_metadatas = prompt_fn(**config.prompt_fn_kwargs)
                global_caption, region_caption_list, region_bboxes_list = transform_meta(prompt_metadatas)

                # print("global_caption:", global_caption)
                # print("region_caption_list:", region_caption_list)
                # print("region_bboxes_list:", region_bboxes_list)

                global_caption_list = global_caption * config.hmn_size
                meta = get_meta(global_caption, region_caption_list, region_bboxes_list, config.resolution)
                metas = tuple(copy.deepcopy(meta) for _ in range(config.hmn_size))

                prompt, layo_prompt, list_cond_image_pil = Hicoinput(global_caption, region_caption_list,
                                                                    region_bboxes_list, img_size=config.resolution)

                images = []
                timesteps_list = []
                data_idxs = []

                for infer in range(config.hmn_size):
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        image, all_latents, all_encoders, all_cross_attention_kwargs, all_down_block_res_samples, all_mid_block_additional_residual, timesteps = pipeline_with_logprob(
                            pipeline,
                            prompt, layo_prompt, guidance_scale=config.sample.guidance_scale, infer_mode=config.sample.mode,
                            num_inference_steps=config.sample.num_steps, image=list_cond_image_pil, fuse_type=config.sample.fuse_type,
                            width=config.resolution, height=config.resolution, s_churn=config.sample.s_churn,
                        )
                     
                    if isinstance(image, list):
                        images.extend(image)
                    else:
                        images.append(image)  
                    
                    if not torch.is_tensor(timesteps):
                        timesteps = torch.as_tensor(timesteps, device=accelerator.device)
                    else:
                        timesteps = timesteps.to(device=accelerator.device)
                    timesteps_list.append(timesteps)        

                    data_idx = store.add(
                        image=image,
                        all_latents=all_latents,
                        all_encoders=all_encoders,
                        all_cross_attention_kwargs=all_cross_attention_kwargs,
                        all_down_block_res_samples=all_down_block_res_samples,
                        all_mid_block_additional_residual=all_mid_block_additional_residual,
                        metadata={"iter": i},
                    )
                    data_idxs.append(data_idx)
                    # print(f"stored data_idx = {data_idx}")

                #################### Hard negative mining ####################
                rewards, metadata = reward_fn(images, global_caption_list, metas)
                rewards_t = torch.as_tensor(rewards, device=accelerator.device)

                top_k = config.hmn_pairs
                bottom_k = config.hmn_pairs
                sorted_idx = torch.argsort(rewards_t, descending=True)
                pos_idx = sorted_idx[:top_k].tolist()
                neg_idx = sorted_idx[-bottom_k:].tolist()        

                for p in pos_idx:
                    for n in neg_idx:
                        pair_indices = torch.tensor([p, n], device=accelerator.device, dtype=torch.long)
                        pair_timesteps = torch.stack([timesteps_list[p], timesteps_list[n]], dim=0)  # (2, T)

                        pair_rewards = rewards_t[pair_indices]  # (2,)
                        pair_data_idxs = torch.tensor([data_idxs[p], data_idxs[n]],
                                                    device=accelerator.device, dtype=torch.long)  # (2,)

                        pair_sample = {
                            "data_idxs":  pair_data_idxs,   # (2,)
                            "timesteps":  pair_timesteps,   # (2, T)
                            "rewards":    pair_rewards,     # (2,)
                        }
                        samples.append(pair_sample)

                        # print("test_paired:", pair_sample)
                        # idx = pair_sample["data_idxs"][0]
                        # print("test_store_latents:", store.get(idx).all_latents[0].shape)
                        # print("test_store_encoders:", store.get(idx).all_encoders[0].shape)
                        # print("test_mid_blocks:", store.get(idx).all_mid_block_additional_residual[0].shape)
                
                #################### PIANTING ####################
                annotate_and_save(images, region_bboxes_list, region_caption_list,
                                "LayoutDPO/saveimg", epoch, i)             

                images.clear()
                timesteps_list.clear()
                data_idxs.clear()

            #################### CHECK ####################
            samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}

            def save_and_prepare_image(image, tmpdir, filename):
                if isinstance(image, Image.Image):
                    pil = image
                else:
                    arr = (image * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
                    pil = Image.fromarray(arr.transpose(1, 2, 0))
                pil = pil.resize((256, 256))
                output_path = os.path.join(tmpdir, filename)
                pil.save(output_path, format="PNG")
                return output_path

            with tempfile.TemporaryDirectory() as tmpdir:
                logged_images = []
                for i, image in enumerate(images):
                    img_path = save_and_prepare_image(image, tmpdir, f"{i}.png")
                    logged_images.append(
                        wandb.Image(
                            img_path,
                            caption=f"{global_caption_list[i]:.25} | {rewards[i]:.2f}"  # only log rewards from process 0
                        )
                    )

                accelerator.log(
                    {
                        "images": logged_images,
                    },
                    step=global_step
                )

            rewards = accelerator.gather(samples["rewards"]).cpu().numpy()

            accelerator.log(
                {
                    "reward": rewards,
                    "epoch": epoch,
                    "reward_mean": rewards.mean(),
                    "reward_std": rewards.std(),
                },
                step=global_step,
            )

            assert len(samples) == 3 # have 3 keys

            #################### TRAINING ####################
            for inner_epoch in range(config.train.num_inner_epochs):
                samples_batched = {k: v.reshape(-1, config.train.batch_size, *v.shape[1:]) for k, v in samples.items()}
                samples_batched = [dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())]
                unet.train()
                unet_ref.eval()
                info = defaultdict(list)

                _p = next(unet.parameters())
                dev = _p.device
                dt  = _p.dtype
                do_cfg = bool(config.train.cfg)

                def ensure_bchw(x: torch.Tensor) -> torch.Tensor:
                    return x if x.dim() == 4 else x.unsqueeze(0)

                for i, sample in tqdm(
                        list(enumerate(samples_batched)),
                        desc=f"Epoch {epoch}.{inner_epoch}: training",
                        position=0,
                        disable=not accelerator.is_local_main_process,
                ):
                    ############# prepare #############
                    winner_idx = sample["data_idxs"][0].item()
                    loser_idx = sample["data_idxs"][1].item()
                    pair_ts = sample["timesteps"].to(dev)
                    rewards = sample["rewards"].to(dev)

                    w_pack = store.get(winner_idx)
                    l_pack = store.get(loser_idx)

                    # set schedulers
                    sched_main = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
                    sched_ref  = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
                    sched_main.set_timesteps(config.sample.num_steps, device=dev)
                    sched_ref.set_timesteps(config.sample.num_steps, device=dev)
            
                    T = pair_ts.shape[1]

                    for j in tqdm(
                            range(T),
                            desc="Timestep",
                            position=1,
                            leave=False,
                            disable=not accelerator.is_local_main_process,
                    ):
                        t = pair_ts[0, j].to(dev)
                        # set timestep
                        sched_main._step_index = None
                        sched_ref._step_index  = None
                        sched_main._init_step_index(t)
                        sched_ref._init_step_index(t)

                        # latents x_t
                        w_lat = ensure_bchw(w_pack.all_latents[j]).to(dev, dtype=dt)
                        l_lat = ensure_bchw(l_pack.all_latents[j]).to(dev, dtype=dt)
                        # text encoder
                        w_enc = w_pack.all_encoders.to(dev, dtype=dt)
                        l_enc = l_pack.all_encoders.to(dev, dtype=dt)
                        # controlnet
                        w_down = [x.to(dev, dtype=dt) for x in w_pack.all_down_block_res_samples[j]]
                        l_down = [x.to(dev, dtype=dt) for x in l_pack.all_down_block_res_samples[j]]
                        w_mid  = w_pack.all_mid_block_additional_residual[j].to(dev, dtype=dt)
                        l_mid  = l_pack.all_mid_block_additional_residual[j].to(dev, dtype=dt)
                        # cross_attention_kwargs
                        ca_kwargs_w = w_pack.all_cross_attention_kwargs[j]
                        ca_kwargs_l = l_pack.all_cross_attention_kwargs[j]
                        ca_kwargs = ca_kwargs_w if ca_kwargs_w is not None else ca_kwargs_l
                        if (ca_kwargs_w is not None) and (ca_kwargs_l is not None):
                            assert ca_kwargs_w == ca_kwargs_l, "the batch cannot be merged."

                        # latents merge
                        if do_cfg:
                            w_in = torch.cat([w_lat, w_lat], dim=0)  # [2, C, H, W]
                            l_in = torch.cat([l_lat, l_lat], dim=0)  # [2, C, H, W]
                        else:
                            w_in, l_in = w_lat, l_lat                 # [1, C, H, W]

                        latent_batched = torch.cat([w_in, l_in], dim=0)         
                        latent_batched = sched_main.scale_model_input(latent_batched, t)

                        # encoder_hidden_states merge
                        enc_batched = torch.cat([w_enc, l_enc], dim=0)            

                        # control merge
                        down_batched = [torch.cat([wd, ld], dim=0) for wd, ld in zip(w_down, l_down)]
                        mid_batched  = torch.cat([w_mid, l_mid], dim=0)            

                        with accelerator.accumulate(unet):
                            with autocast():
                                noise_pred_batch = unet(
                                    latent_batched,
                                    t,
                                    encoder_hidden_states=enc_batched,
                                    cross_attention_kwargs=ca_kwargs,
                                    down_block_additional_residuals=down_batched,
                                    mid_block_additional_residual=mid_batched,
                                    return_dict=False,
                                )[0]

                                with torch.no_grad():
                                    noise_pred_ref_batch = unet_ref(
                                        latent_batched,
                                        t,
                                        encoder_hidden_states=enc_batched,
                                        cross_attention_kwargs=ca_kwargs,
                                        down_block_additional_residuals=down_batched,
                                        mid_block_additional_residual=mid_batched,
                                        return_dict=False,
                                    )[0]

                                if do_cfg:
                                    w_u, w_c, l_u, l_c = torch.split(noise_pred_batch, 1, dim=0)
                                    w_pred = w_u + config.sample.guidance_scale * (w_c - w_u)
                                    l_pred = l_u + config.sample.guidance_scale * (l_c - l_u)

                                    w_u_r, w_c_r, l_u_r, l_c_r = torch.split(noise_pred_ref_batch, 1, dim=0)
                                    w_pred_ref = w_u_r + config.sample.guidance_scale * (w_c_r - w_u_r)
                                    l_pred_ref = l_u_r + config.sample.guidance_scale * (l_c_r - l_u_r)
                                else:
                                    w_pred, l_pred = torch.split(noise_pred_batch, 1, dim=0)
                                    w_pred_ref, l_pred_ref = torch.split(noise_pred_ref_batch, 1, dim=0)

                            # Calculate log probabilities
                            x_t_batch = torch.cat([ensure_bchw(w_lat), ensure_bchw(l_lat)], dim=0)
                            pred_batch = torch.cat([w_pred, l_pred], dim=0)
                            pred_ref_batch = torch.cat([w_pred_ref, l_pred_ref], dim=0)

                            _, log_prob = dpm_step_with_logprob(
                                sched_main, pred_batch,
                                t, x_t_batch, s_churn=config.sample.s_churn
                            )
                            _, log_prob_ref = dpm_step_with_logprob(
                                sched_ref, pred_ref_batch,
                                t, x_t_batch, s_churn=config.sample.s_churn)

                            if config.train.method == "dpso":
                                reward_diff = sample["rewards"][0] - sample["rewards"][1]
                                w_r = 1.0 - torch.sigmoid(reward_diff)

                                lam = config.train.dpso_lmbda
                                gamma = config.train.dpso_gamma
                                adjusted_ref = lam * gamma * w_r
                                # print("adjusted_ref", adjusted_ref)

                                # print("test_logprob:", log_prob)
                                reward_diff = sample['rewards'][:, None] - sample['rewards']
                                # print("test:",reward_diff)
                                ratio = torch.log(
                                    torch.clamp(torch.exp(log_prob - log_prob_ref), 1 - config.train.eps, 1 + config.train.eps))
                                # print("ratio=:", ratio)
                                ratio_diff = ratio[:, None] - ratio
                                # print("ratio_diff:", ratio_diff)
                                loss = -torch.log(torch.sigmoid(adjusted_ref * ratio_diff * (reward_diff > 0))).mean()
                                # print("test_inner:",config.train.beta * ratio_diff * (reward_diff > 0))
                                print(loss)
                            else:
                                reward_mat = rewards[:, None] - rewards[None, :]
                                ratio = (log_prob - log_prob_ref).exp().clamp(1 - config.train.dpo_eps, 1 + config.train.dpo_eps).log()
                                ratio_diff = ratio.unsqueeze(1) - ratio
                                mask = (reward_mat > 0).float()
                                scaled = config.train.dpo_beta * ratio_diff * mask
                                loss = -torch.log(torch.sigmoid(scaled)).mean()
                                print(loss)

                            info["loss"].append(loss)
                            # backward pass
                            accelerator.backward(loss)
                            if accelerator.sync_gradients:
                                accelerator.clip_grad_norm_(
                                    unet.parameters(), config.train.max_grad_norm
                                )
                            optimizer.step()
                            optimizer.zero_grad()

                        # Checks if the accelerator has performed an optimization step behind the scenes
                        if accelerator.sync_gradients:
                            assert (j == T - 1) and (
                                    i + 1
                            ) % config.train.gradient_accumulation_steps == 0
                            # log training-related stuff
                            info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                            info = accelerator.reduce(info, reduction="mean")
                            info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                            accelerator.log(info, step=global_step)
                            global_step += 1
                            info = defaultdict(list)

                # make sure we did an optimization step at the end of the inner epoch
                assert accelerator.sync_gradients

            if epoch != 0 and epoch % config.save_freq == 0 and accelerator.is_main_process:
                accelerator.save_state(os.path.join(os.path.join(config.logdir, config.run_name), f"checkpoint_{epoch}"))


if __name__ == "__main__":
    app.run(main)