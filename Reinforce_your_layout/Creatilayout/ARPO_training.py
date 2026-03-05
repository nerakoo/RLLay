import os
import copy
import tempfile
import contextlib
import datetime
from collections import defaultdict
from functools import partial
from concurrent import futures
import torch.nn.functional as F
import time
import random
import numpy as np

import torch
import wandb
from PIL import Image
from absl import app, flags
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_unet_state_dict_to_peft
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.training_utils import compute_density_for_timestep_sampling
from ml_collections import config_flags
from peft import LoraConfig, set_peft_model_state_dict
from peft import get_peft_model, PeftModel
from peft.utils import get_peft_model_state_dict
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Sampler

import utils.prompts
import utils.rewards
from utils.utils import transform_iter_meta, get_meta, annotate_and_save_with_rank, normalize_rewards_to_dict, DummyFuture, PerPromptStatTracker
# from training_patch.pipeline.sd3_encoder import encode_prompt
from training_patch.pipeline.sd3_arpo_pipeline_with_logprob import sd3_pipeline_with_logprob
from src.models.transformer_sd3_SiamLayout import SiamLayoutSD3Transformer2DModel
from src.pipeline.pipeline_sd3_CreatiLayout import CreatiLayoutSD3Pipeline
from training_patch.ema import EMAModuleWrapper
from utils.prompts_dataloader import LayoutJSONPromptDataset
from utils.store_util import LayoutKwargsStore

tqdm = partial(tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base_sd3.py", "Training configuration.")
logger = get_logger(__name__)

# ========= Memory helpers =========
def enable_sdpa_memory_efficient():
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
    except Exception:
        pass

@contextlib.contextmanager
def offload_te_and_vae_during_training(pipeline):
    devs = {
        "vae": getattr(pipeline.vae, "device", torch.device("cpu")),
        "te1": getattr(pipeline.text_encoder, "device", torch.device("cpu")),
        "te2": getattr(pipeline.text_encoder_2, "device", torch.device("cpu")),
        "te3": getattr(pipeline.text_encoder_3, "device", torch.device("cpu")),
    }
    try:
        pipeline.vae.to("cpu")
        pipeline.text_encoder.to("cpu")
        pipeline.text_encoder_2.to("cpu")
        pipeline.text_encoder_3.to("cpu")
        torch.cuda.empty_cache()
        yield
    finally:
        pipeline.vae.to(devs["vae"])
        pipeline.text_encoder.to(devs["te1"])
        pipeline.text_encoder_2.to(devs["te2"])
        pipeline.text_encoder_3.to(devs["te3"])

def maybe_enable_grad_checkpointing(module):
    for attr in ["enable_gradient_checkpointing", "gradient_checkpointing_enable"]:
        if hasattr(module, attr):
            try:
                getattr(module, attr)()
            except Exception:
                pass
            break

def log_cuda_mem(prefix=""):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        used = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"[{prefix}] CUDA used={used:.0f}MB, reserved={reserved:.0f}MB")

class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size  # Batch size per replica
        self.k = k                    # Number of repetitions per sample
        self.num_replicas = num_replicas  # Total number of replicas
        self.rank = rank              # Current replica rank
        self.seed = seed              # Random seed for synchronization
        
        # Compute the number of unique samples needed per iteration
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, f"k can not divide n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k  # Number of unique samples
        self.epoch = 0

    def __iter__(self):
        while True:
            # Generate a deterministic random sequence to ensure all replicas are synchronized
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            
            # Randomly select m unique samples
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            
            # Repeat each sample k times to generate n*b total samples
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            
            # Shuffle to ensure uniform distribution
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            
            # Split samples to each replica
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            
            # Return current replica's sample indices
            yield per_card_samples[self.rank]
    
    def set_epoch(self, epoch):
        self.epoch = epoch  # Used to synchronize random state across epochs

def get_sigmas(noise_scheduler, timesteps, accelerator, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
    timesteps = timesteps.to(accelerator.device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma

def copy_learner_to_ref(transformer):
    for name, param in transformer.named_parameters():
        if "learner" in name:
            ref_name = name.replace("learner", "ref")
            ref_param = dict(transformer.named_parameters())[ref_name]
            ref_param.data.copy_(param.data)

def calculate_zero_std_ratio(prompts, gathered_rewards):
    # Convert prompt list to NumPy array
    prompt_array = np.array(prompts)
    
    # Get unique prompts and their group information
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, 
        return_inverse=True,
        return_counts=True
    )
    
    # Group rewards for each prompt
    grouped_rewards = gathered_rewards['ori_avg'][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    
    # Calculate standard deviation for each group
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    
    # Calculate the ratio of zero standard deviation
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    
    return zero_std_ratio, prompt_std_devs.mean()

def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def save_ckpt(save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config):
    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    save_root_lora = os.path.join(save_root, "lora")
    os.makedirs(save_root_lora, exist_ok=True)
    if accelerator.is_main_process:
        if config.train.ema:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        unwrap_model(transformer, accelerator).base_model.model.save_pretrained(save_root_lora)
        if config.train.ema:
            ema.copy_temp_to(transformer_trainable_parameters)

def main(_):
    enable_sdpa_memory_efficient()
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

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
            project_name="fine-tune-sd3-dspo",
            config=config.to_dict(),
            init_kwargs={"wandb": {"name": config.run_name}},
        )
    logger.info(f"\n{config}")

    set_seed(config.seed, device_specific=True)

    transformer_additional_kwargs = dict(attention_type="layout", strict=True)
    transformer = SiamLayoutSD3Transformer2DModel.from_pretrained(
        config.pretrained.transformer, subfolder="SiamLayout_SD3", torch_dtype=torch.bfloat16,
        **transformer_additional_kwargs)
    pipeline = CreatiLayoutSD3Pipeline.from_pretrained(config.pretrained.base_model, transformer=transformer,
                                                       torch_dtype=torch.bfloat16).to(accelerator.device)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)

    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]

    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    pipeline.vae.to(accelerator.device, dtype=torch.float32)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_3.to(accelerator.device, dtype=inference_dtype)
    pipeline.transformer.to(accelerator.device)

    if config.use_lora:
        target_modules = [
            "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
            "attn.to_k", "attn.to_out.0", "attn.to_q", "attn.to_v",
        ]
        transformer_lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_rank,
            init_lora_weights="gaussian",
            target_modules=target_modules
        )
        if config.train.lora_path:
            pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, config.train.lora_path)
            # After loading with PeftModel.from_pretrained, all parameters have requires_grad set to False. You need to call set_adapter to enable gradients for the adapter parameters.
            pipeline.transformer.set_adapter("default")
        else:
            pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config, adapter_name="learner")
            pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config, adapter_name="ref")
            pipeline.transformer.set_adapter("learner")

    transformer = pipeline.transformer
    # maybe_enable_grad_checkpointing(transformer)
    transformer_trainable_parameters = []
    for name, param in transformer.named_parameters():
        if "learner" in name:
            assert param.requires_grad == True
            transformer_trainable_parameters.append(param)
    
    ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8, device=accelerator.device)

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

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

    optimizer = optimizer_cls(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # prepare prompt and reward fn
    prompt_fn = getattr(utils.prompts, config.prompt_fn)
    reward_fn = getattr(utils.rewards, config.reward_fn)(
        device=accelerator.device)  # ensure reward model is on same device

    # prepare dataset
    train_dataset = LayoutJSONPromptDataset(
        json_path=config.prompt_path,
        final_w=config.resolution,
        final_h=config.resolution,
    )

    train_sampler = DistributedKRepeatSampler( 
        dataset=train_dataset,
        batch_size=config.sample.train_batch_size,
        k=config.sample.num_image_per_prompt,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        seed=config.seed
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=1,
        collate_fn=LayoutJSONPromptDataset.collate_fn,
    )

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    # initialize stat tracker
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)

    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    transformer, optimizer, train_dataloader = accelerator.prepare(transformer, optimizer, train_dataloader)
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        config.pretrained.base_model, subfolder="scheduler"
    )
    executor = futures.ThreadPoolExecutor(max_workers=8)

    # Train!
    samples_per_epoch = (
        config.sample.train_batch_size
        * accelerator.num_processes
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    # assert config.sample.batch_size == config.train.batch_size 
    # assert config.sample.batch_size == 2  # for dpo loss
    # assert config.hmn_size >= config.hmn_pairs  # for HNM
    # assert config.train.gradient_accumulation_steps == config.hmn_pairs * config.hmn_pairs * config.sample.num_batches_per_epoch

    #######################test######################################
    # import matplotlib.pyplot as plt
    # batch_size = 4
    # num_inference_steps = 50
    # guidance_scale = 7.5
    # height = 1024
    # width = 1024
    #
    # global_caption=[
    #     'This is a photo showcasing a mosque with a typical Islamic architectural style. The mosque is characterized by its towering minaret and large dome, both of which are made of stone. The entrance of the mosque is a double-arched porch, with a row of decorative pillars on both sides. The exterior walls of the building are made of stone, with a grey tone, and the roof is covered with a grey dome. The surrounding environment is a spacious square, with a few trees and some low-rise buildings visible in the background.']
    # region_caption_list=['trees', 'trees', 'minaret', 'large dome grey dome', 'spacious square']
    # region_bboxes_list=[[0.769603438013337, 0.4799966227213542, 0.9985172047334558, 0.920195556640625],
    #                      [0.0008651763387257525, 0.603338134765625, 0.19105535514363658, 0.9425865071614583],
    #                      [0.675440323964526, 0.06483316548665365, 0.7717202655663329, 0.6422011311848959],
    #                      [0.3071458074118527, 0.48534086100260415, 0.6709005000146667, 0.6512798258463541],
    #                      [0.00197142355134699, 0.9511194661458333, 0.9976715737200798, 0.9983336588541667]]
    #
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # pipeline = pipeline.to(device)
    # pipeline.vae.to(torch.float32)
    # with torch.no_grad():
    #     images = pipeline(prompt=global_caption * batch_size,
    #                   generator=torch.Generator(device=device).manual_seed(42),
    #                   num_inference_steps=num_inference_steps,
    #                   guidance_scale=guidance_scale,
    #                   bbox_phrases=region_caption_list,
    #                   bbox_raw=region_bboxes_list,
    #                   height=height,
    #                   width=width
    #                   )
    # images = images.images
    #
    # pil_img = images[0]
    # save_dir = "path_to_test_output_dir"
    # os.makedirs(save_dir, exist_ok=True)
    # save_path = os.path.join(save_dir, "output.png")
    # assert isinstance(pil_img, Image.Image), f"Expected PIL.Image, got {type(pil_img)}"
    # pil_img.save(save_path)
    # print("Saved to:", save_path, "Exists:", os.path.exists(save_path))
    #
    # pil_img.show()
    # print(pil_img)
    # print("111")
    ################################################################
    epoch = 0
    global_step = 0
    train_iter = iter(train_dataloader)

    while True:
        #################### SAVE ####################
        pipeline.transformer.eval()
        if epoch % config.save_freq == 0 and epoch > 0 and accelerator.is_main_process:
            save_ckpt(config.save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config)

        #################### SAMPLING ####################
        pipeline.transformer.eval()
        samples = []
        for i in tqdm(
                range(config.sample.num_batches_per_epoch),
                desc=f"Epoch {epoch}: sampling",
                disable=not accelerator.is_local_main_process,
                position=0,
        ):
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            prompts, prompt_metadatas = next(train_iter) # inference "config.sample.train_batch_size" images
            global_captions, region_captions_list, region_bboxes_list = transform_iter_meta(prompt_metadatas)
            # print("global_captions:", global_captions)
            # print("region_caption_list:", region_caption_list)
            # print("region_bboxes_list:", region_bboxes_list)
            metas = get_meta(global_captions, region_captions_list, region_bboxes_list)
            # print("metas:", metas)

            prompt_ids = tokenizers[0](
                prompts,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(accelerator.device)
            # print("prompt_ids_shape", prompt_ids.shape)

            if global_step>0 and global_step%config.train.ref_update_step==0:
                copy_learner_to_ref(transformer)

            images_list = []
            latents_last_list = []
            log_probs_list = []
            ret_ts_list = []
            prompt_embeds_list = []
            pooled_prompt_embeds_list = []
            boxes_list = []
            text_embeds_list = []
            masks_list = []

            with autocast():
                with torch.no_grad():
                    pipeline.transformer.set_adapter("ref")
                    for idx in range(len(global_captions)):
                        img_i, latents_i, logp_i, ret_ts_i, pemb_i, ppemb_i, boxes_i, text_emb_i, masks_i = sd3_pipeline_with_logprob(
                            pipeline,
                            prompt=[global_captions[idx]],         
                            num_inference_steps=config.sample.num_steps,
                            guidance_scale=config.sample.guidance_scale,
                            bbox_phrases=region_captions_list[idx],    
                            bbox_raw=region_bboxes_list[idx],          
                            s_churn=config.sample.s_churn,
                            output_type="pt",
                            height=config.resolution,
                            width=config.resolution,
                        )

                        images_list.append(img_i)                     # [1, 3, H, W]
                        latents_last_list.append(latents_i[-1])       # [1, C_lat, H_lat, W_lat]
                        log_probs_list.append(logp_i)                
                        ret_ts_list.append(ret_ts_i)                 
                        prompt_embeds_list.append(pemb_i)             # [1, L, D_attn]
                        pooled_prompt_embeds_list.append(ppemb_i)     # [1, D_pool]
                        boxes_list.append(boxes_i)                    # [1, Mi, 4]
                        text_embeds_list.append(text_emb_i)           # [1, Mi, D_txt]
                        masks_list.append(masks_i)

            import torch.nn.functional as F
            def _pad_dim1_to(t, target_len):
                if t.size(1) == target_len:
                    return t
                pad_len = target_len - t.size(1)
                pad_shape = list(t.shape)
                pad_shape[1] = pad_len
                pad_tensor = t.new_zeros(pad_shape)
                return torch.cat([t, pad_tensor], dim=1)

            images = torch.cat(images_list, dim=0).to(torch.float32)                   # [B, 3, H, W]
            latents_last = torch.cat(latents_last_list, dim=0).to(torch.float32)       # [B, C_lat, H_lat, W_lat]
            prompt_embeds = torch.cat(prompt_embeds_list, dim=0).to(torch.float32)     # [B, L, D_attn]
            pooled_prompt_embeds = torch.cat(pooled_prompt_embeds_list, dim=0).to(torch.float32)  # [B, D_pool]
            boxes = torch.cat(boxes_list, dim=0).to(torch.float32)                 # [B, MAX_REGIONS, 4]
            text_embeddings = torch.cat(text_embeds_list, dim=0).to(torch.float32) # [B, MAX_REGIONS, D_txt]
            masks = torch.cat(masks_list, dim=0).to(torch.float32)                 # [B, MAX_REGIONS, Hm, Wm]

            rewards, metadata = reward_fn(images, global_captions, metas)
            time.sleep(0)

            rewards_dict = normalize_rewards_to_dict(rewards)
            fut = DummyFuture(rewards_dict, metadata)

            # print("boxes_shape:", return_boxes.shape)
            # print("text_embeddings_shape:", return_text_embeddings.shape)
            # print("masks_shape:", return_masks.shape)

            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "latents": latents_last,
                    "rewards": fut,
                    "boxes": boxes,
                    "text_embeddings": text_embeddings,
                    "masks": masks,
                }
            )

            #################### PIANTING ####################
            rank = accelerator.process_index
            annotate_and_save_with_rank(images, region_bboxes_list, region_captions_list,
                              "LayoutDPO/saveimg", epoch, i, k=rank)

            torch.cuda.empty_cache()

        #################### CHECK ####################
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"].result()
            sample["rewards"] = {
                key: torch.as_tensor(value, device=accelerator.device).float()
                for key, value in rewards.items()
            }

        samples = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {
                sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                for sub_key in samples[0][k]
            }
            for k in samples[0].keys()
        }

        if epoch % 10 == 0 and accelerator.is_main_process:
            with tempfile.TemporaryDirectory() as tmpdir:
                num_samples = min(15, len(images))
                sample_indices = random.sample(range(len(images)), num_samples)

                for idx, i in enumerate(sample_indices):
                    image = images[i]
                    pil = Image.fromarray(
                        (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))  

                sampled_prompts = [prompts[i] for i in sample_indices]
                sampled_rewards = [rewards['avg'][i] for i in sample_indices]

                wandb.log(
                    {
                        "images": [
                            wandb.Image(
                                os.path.join(tmpdir, f"{idx}.jpg"),
                                caption=f"{prompt:.100} | avg: {avg_reward:.2f}",
                            )
                            for idx, (prompt, avg_reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                        ],
                    },
                    step=global_step,
                )
        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]
        samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(-1)

        # gather rewards across processes
        gathered_rewards = {key: accelerator.gather(value) for key, value in samples["rewards"].items()}
        gathered_rewards = {key: value.cpu().numpy() for key, value in gathered_rewards.items()}
        # log rewards and images
        if accelerator.is_main_process:
            wandb.log(
                {
                    "epoch": epoch,
                    **{f"reward_{key}": value.mean() for key, value in gathered_rewards.items() if '_strict_accuracy' not in key and '_accuracy' not in key},
                },
                step=global_step,
            )

        # per-prompt mean/std tracking
        if config.per_prompt_stat_tracking:
            # gather the prompts across processes
            prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompts = pipeline.tokenizer.batch_decode(
                prompt_ids, skip_special_tokens=True
            )
            advantages = stat_tracker.update(prompts, gathered_rewards['avg'], type=config.train.algorithm)
            if accelerator.is_local_main_process:
                print("len(prompts)", len(prompts))
                print("len unique prompts", len(set(prompts)))

            group_size, trained_prompt_num = stat_tracker.get_stats()

            zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts, gathered_rewards)

            if accelerator.is_main_process:
                wandb.log(
                    {
                        "group_size": group_size,
                        "trained_prompt_num": trained_prompt_num,
                        "zero_std_ratio": zero_std_ratio,
                        "reward_std_mean": reward_std_mean,
                    },
                    step=global_step,
                )
            stat_tracker.clear()
        else:
            advantages = (gathered_rewards['avg'] - gathered_rewards['avg'].mean()) / (gathered_rewards['avg'].std() + 1e-4)

        prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
        latents = accelerator.gather(samples['latents']).cpu().numpy()
        prompt_embeds = accelerator.gather(samples['prompt_embeds']).cpu().numpy()
        pooled_prompt_embeds = accelerator.gather(samples['pooled_prompt_embeds']).cpu().numpy()
        boxes = accelerator.gather(samples["boxes"]).cpu().numpy()
        text_embeddings = accelerator.gather(samples["text_embeddings"]).cpu().numpy()
        masks = accelerator.gather(samples["masks"]).cpu().numpy()
        
        # Filter out samples with non-zero advantages
        non_zero_indices = np.where(advantages != 0)[0]
        filtered_advantages = advantages[non_zero_indices]
        filtered_latents = latents[non_zero_indices]
        filtered_prompt_ids = prompt_ids[non_zero_indices]
        filtered_prompt_embeds = prompt_embeds[non_zero_indices]
        filtered_pooled_prompt_embeds = pooled_prompt_embeds[non_zero_indices]
        filtered_boxes = boxes[non_zero_indices]
        filtered_text_embeddings = text_embeddings[non_zero_indices]
        filtered_masks = masks[non_zero_indices]

        # Group latents by prompt_ids
        unique_prompt_ids = np.unique(filtered_prompt_ids, axis=0)
        concat_advantages = []
        concat_latent = []
        concat_prompt_embeds = []
        concat_pooled_prompt_embeds = []
        concat_boxes = []
        concat_text_embeddings = []
        concat_masks = []

        for prompt_id in unique_prompt_ids:
            # Find indices where prompt_id matches
            matches = np.where(np.all(filtered_prompt_ids == prompt_id, axis=1))[0]
            advantages = filtered_advantages[matches]
            latents = filtered_latents[matches]
            prompt_embeds = filtered_prompt_embeds[matches]
            pooled_prompt_embeds = filtered_pooled_prompt_embeds[matches]
            boxes = filtered_boxes[matches]
            text_embeddings = filtered_text_embeddings[matches]
            masks = filtered_masks[matches]

            concat_advantages.append(advantages)
            concat_latent.append(latents)
            concat_prompt_embeds.append(prompt_embeds)
            concat_pooled_prompt_embeds.append(pooled_prompt_embeds)
            concat_boxes.append(boxes)
            concat_text_embeddings.append(text_embeddings)
            concat_masks.append(masks)
        
        # Stack all grouped latents
        concat_advantages = np.stack(concat_advantages, axis=0)  # Shape: [num_prompts, 2, 1]
        concat_latent = np.stack(concat_latent, axis=0)  # Shape: [num_prompts, 2, 16, 64, 64]
        concat_prompt_embeds = np.stack(concat_prompt_embeds, axis=0)
        concat_pooled_prompt_embeds = np.stack(concat_pooled_prompt_embeds, axis=0)
        concat_boxes = np.stack(concat_boxes, axis=0)
        concat_text_embeddings = np.stack(concat_text_embeddings, axis=0)
        concat_masks = np.stack(concat_masks, axis=0)

        concat_advantages = torch.as_tensor(concat_advantages)
        concat_latent = torch.as_tensor(concat_latent)
        concat_prompt_embeds = torch.as_tensor(concat_prompt_embeds)
        concat_pooled_prompt_embeds = torch.as_tensor(concat_pooled_prompt_embeds)
        concat_boxes = torch.as_tensor(concat_boxes)
        concat_text_embeddings = torch.as_tensor(concat_text_embeddings)
        concat_masks = torch.as_tensor(concat_masks)

        # This is because when handling multiple tasks, our prompt dataset contains duplicates, which leads to inconsistent group sizes.
        # Check if we have enough samples to distribute across processes
        min_required_samples = accelerator.num_processes
        current_samples = concat_advantages.shape[0]
        
        # If we don't have enough samples, randomly sample from existing ones to make it divisible
        if current_samples % min_required_samples != 0:
            samples_needed = min_required_samples - (current_samples % min_required_samples)
            # Randomly select indices to duplicate
            random_indices = torch.randint(0, current_samples, (samples_needed,))
            
            # Append the randomly sampled data to make it divisible by num_processes
            concat_advantages = torch.cat([concat_advantages, concat_advantages[random_indices]], dim=0)
            concat_latent = torch.cat([concat_latent, concat_latent[random_indices]], dim=0)
            concat_prompt_embeds = torch.cat([concat_prompt_embeds, concat_prompt_embeds[random_indices]], dim=0)
            concat_pooled_prompt_embeds = torch.cat([concat_pooled_prompt_embeds, concat_pooled_prompt_embeds[random_indices]], dim=0)
            concat_boxes = torch.cat([concat_boxes, concat_boxes[random_indices]], dim=0)
            concat_text_embeddings = torch.cat([concat_text_embeddings, concat_text_embeddings[random_indices]], dim=0)
            concat_masks = torch.cat([concat_masks, concat_masks[random_indices]], dim=0)

        advantages = concat_advantages.reshape(accelerator.num_processes, -1, *concat_advantages.shape[1:])[accelerator.process_index].to(accelerator.device)
        latents = concat_latent.reshape(accelerator.num_processes, -1, *concat_latent.shape[1:])[accelerator.process_index].to(accelerator.device)
        prompt_embeds = concat_prompt_embeds.reshape(accelerator.num_processes, -1, *concat_prompt_embeds.shape[1:])[accelerator.process_index].to(accelerator.device)
        pooled_prompt_embeds = concat_pooled_prompt_embeds.reshape(accelerator.num_processes, -1, *concat_pooled_prompt_embeds.shape[1:])[accelerator.process_index].to(accelerator.device)
        boxes = concat_boxes.reshape(accelerator.num_processes, -1, *concat_boxes.shape[1:])[accelerator.process_index].to(accelerator.device)
        text_embeddings = concat_text_embeddings.reshape(accelerator.num_processes, -1, *concat_text_embeddings.shape[1:])[accelerator.process_index].to(accelerator.device)
        masks = concat_masks.reshape(accelerator.num_processes, -1, *concat_masks.shape[1:])[accelerator.process_index].to(accelerator.device)

        advantages = advantages.squeeze(-1)      
        num_prompts = advantages.shape[0]

        for i in range(num_prompts):
            if advantages[i, 0] == -1:
                temp = latents[i, 0].clone()
                latents[i, 0] = latents[i, 1].clone()
                latents[i, 1] = temp

        latents = latents.permute(1, 0, 2, 3, 4)                        # [2, N, C, H, W]
        prompt_embeds = prompt_embeds.permute(1, 0, 2, 3)               # [2, N, S, D]
        pooled_prompt_embeds = pooled_prompt_embeds.permute(1, 0, 2)    # [2, N, Dp]
        boxes = boxes.permute(1, 0, 2, 3)              # [2, N, M, 4]
        text_embeddings = text_embeddings.permute(1, 0, 2, 3)  # [2, N, M, 2048]
        masks = masks.permute(1, 0, 2) 
        
        samples["latents"] = latents.reshape(-1, *latents.shape[2:])                         # [2N, C, H, W]
        samples["prompt_embeds"] = prompt_embeds.reshape(-1, *prompt_embeds.shape[2:])       # [2N, S, D]
        samples["pooled_prompt_embeds"] = pooled_prompt_embeds.reshape(-1, *pooled_prompt_embeds.shape[2:])  # [2N, Dp]
        samples["boxes"] = boxes.reshape(-1, *boxes.shape[2:])                    # [2N, B, 4]
        samples["text_embeddings"] = text_embeddings.reshape(-1, *text_embeddings.shape[2:])       # [2N, D]
        samples["masks"] = masks.reshape(-1, *masks.shape[2:])                   # [2N, B]
        
        del samples["rewards"]
        del samples["prompt_ids"]

        total_batch_size = len(samples["latents"])
        num_timesteps = config.sample.num_steps

        # #################### TRAINING ####################
        pipeline.transformer.set_adapter("learner")
        for inner_epoch in range(config.train.num_inner_epochs):
            # rebatch for training
            samples_batched = {
                k: v.reshape(-1, total_batch_size//config.sample.num_batches_per_epoch, *v.shape[1:])
                for k, v in samples.items()
            }

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())]

            pipeline.transformer.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                    list(enumerate(samples_batched)),
                    desc=f"Epoch {epoch}.{inner_epoch}: training",
                    position=0,
                    disable=not accelerator.is_local_main_process,
            ):
                bbox_scale = 1.0
                num_grounding_steps = int(0.3 * config.sample.num_steps)

                embeds = sample["prompt_embeds"]
                pooled_embeds = sample["pooled_prompt_embeds"]
                boxes = sample["boxes"]
                text_embeddings = sample["text_embeddings"]
                masks = sample["masks"]

                layout_kwargs = {
                    "layout": {"boxes": boxes, "positive_embeddings": text_embeddings, "masks": masks}
                }

                train_timesteps = [step_index  for step_index in range(num_train_timesteps)]

                for j in tqdm(
                        train_timesteps,
                        desc="Timestep",
                        position=1,
                        leave=False,
                        disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(transformer):
                        if j == num_grounding_steps:
                            bbox_scale = 0.0

                        model_input = sample["latents"]
                        bsz = model_input.shape[0] // 2
                        noise = torch.randn_like(model_input)
                        noise = torch.cat([noise[:bsz], noise[:bsz]], dim=0)

                        u = compute_density_for_timestep_sampling(
                            weighting_scheme='logit_normal',
                            batch_size=bsz,
                            logit_mean=0,
                            logit_std=1,
                            mode_scale=1.29,
                        )

                        indices = (u * noise_scheduler.config.num_train_timesteps).long()
                        timesteps = noise_scheduler.timesteps[indices].to(device=model_input.device)
                        timesteps = torch.cat([timesteps, timesteps], dim=0)
                        sigmas = get_sigmas(noise_scheduler, timesteps, accelerator, n_dim=model_input.ndim, dtype=model_input.dtype)
                        noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                        with autocast():
                            pipeline.transformer.set_adapter("learner")
                            model_pred = transformer(
                                hidden_states=noisy_model_input,
                                timestep=timesteps,
                                encoder_hidden_states=embeds,
                                pooled_projections=pooled_embeds,
                                return_dict=False,
                                layout_kwargs=layout_kwargs,
                                bbox_scale=bbox_scale
                            )[0]
                            with torch.no_grad():
                                pipeline.transformer.set_adapter("ref")
                                model_pred_ref = transformer(
                                    hidden_states=noisy_model_input,
                                    timestep=timesteps,
                                    encoder_hidden_states=embeds,
                                    pooled_projections=pooled_embeds,
                                    return_dict=False,
                                    layout_kwargs=layout_kwargs,
                                    bbox_scale=bbox_scale
                                )[0]
                                model_pred_ref = model_pred_ref.detach()
                                pipeline.transformer.set_adapter("learner")
                        target = noise - model_input

                        model_diff_w, model_diff_l = (model_pred - target).chunk(2, dim=0)
                        model_losses_w = model_diff_w.pow(2).mean(dim=[1, 2, 3])
                        model_losses_l = model_diff_l.pow(2).mean(dim=[1, 2, 3])

                        raw_model_loss = 0.5 * (model_losses_w.mean() + model_losses_l.mean())
                        model_diff = model_losses_w - model_losses_l

                        with torch.no_grad():
                            ref_losses = (model_pred_ref - target).pow(2).mean(dim=[1, 2, 3])
                            ref_losses_w, ref_losses_l = ref_losses.chunk(2, dim=0)
                            ref_diff = ref_losses_w - ref_losses_l
                            raw_ref_loss = ref_losses.mean()

                        scale_term = -0.5 * config.train.beta_dspo
                        inside_term = scale_term * (model_diff - ref_diff)

                        implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)
                        pred2, _ = (model_pred - model_pred_ref).chunk(2, dim=0)
                        loss = (model_diff_w - config.train.beta_dspo * (1 - F.sigmoid(inside_term))[:, None, None, None] * pred2).pow(2).mean(dim=[1, 2, 3]).mean()
                        
                        info["loss"].append(loss)
                        info["raw_model_loss"].append(torch.mean(raw_model_loss))
                        info["implicit_acc"].append(torch.mean(implicit_acc))
                        # backward pass
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                transformer.parameters(), config.train.max_grad_norm
                            )
                        optimizer.step()
                        optimizer.zero_grad()

                    if accelerator.sync_gradients:
                        # assert (j == train_timesteps[-1]) and (
                        #     i + 1
                        # ) % config.train.gradient_accumulation_steps == 0
                        # log training-related stuff
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        if accelerator.is_main_process:
                            wandb.log(info, step=global_step)
                        info = defaultdict(list)
                        global_step += 1
                        if config.train.ema:
                            ema.step(transformer_trainable_parameters, global_step)

        epoch+=1    

if __name__ == "__main__":
    app.run(main)
