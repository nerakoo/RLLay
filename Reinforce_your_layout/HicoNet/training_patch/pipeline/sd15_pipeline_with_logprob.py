import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from src.HiCo_T2I.diffusers.src.diffusers.image_processor import PipelineImageInput
from src.HiCo_T2I.diffusers.src.diffusers.pipelines.controlnet.pipeline_hiconet_layout import ControlNetModel, StableDiffusionHicoNetLayoutPipeline
from training_patch.logprob.dpm_step_with_logprob import dpm_step_with_logprob
from src.HiCo_T2I.diffusers.src.diffusers.utils import (
    is_compiled_module,
)
from src.HiCo_T2I.diffusers.src.diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel

def _to_cpu(x, dtype=None):
    if isinstance(x, torch.Tensor):
        y = x.detach().to("cpu")
        if dtype is not None:
            y = y.to(dtype)
        return y
    if isinstance(x, (list, tuple)):
        return type(x)(_to_cpu(xx, dtype) for xx in x)
    if isinstance(x, dict):
        return {k: _to_cpu(v, dtype) for k, v in x.items()}
    return x

@torch.no_grad()
def pipeline_with_logprob(
    self: StableDiffusionHicoNetLayoutPipeline,
    prompt: Union[str, List[str]] = None,
    layo_prompt: Union[str, List[str]] = None,
    # layo_cond: Union[torch.FloatTensor] = None,
    fuse_type: str = 'sum',
    infer_mode: str = 'single',
    image: PipelineImageInput = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: int = 1,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
    guess_mode: bool = False,
    control_guidance_start: Union[float, List[float]] = 0.0,
    control_guidance_end: Union[float, List[float]] = 1.0,
    s_churn: float = 0.1,
):
    controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

    # align format for control guidance
    if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
        control_guidance_start = len(control_guidance_end) * [control_guidance_start]
    elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
        control_guidance_end = len(control_guidance_start) * [control_guidance_end]
    elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
        mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
        control_guidance_start, control_guidance_end = mult * [control_guidance_start], mult * [
            control_guidance_end
        ]

    # pdb.set_trace()
    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        image,
        callback_steps,
        negative_prompt,
        prompt_embeds,
        negative_prompt_embeds,
        controlnet_conditioning_scale,
        control_guidance_start,
        control_guidance_end,
    )
    # pdb.set_trace()

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device
    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale >= 1.0

    if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
        controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

    global_pool_conditions = (
        controlnet.config.global_pool_conditions
        if isinstance(controlnet, ControlNetModel)
        else controlnet.nets[0].config.global_pool_conditions
    )
    guess_mode = guess_mode or global_pool_conditions

    # 3. Encode input prompt
    text_encoder_lora_scale = (
        cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
    )
    prompt_embeds, negative_prompt_embeds = self.encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        lora_scale=text_encoder_lora_scale,
    )
    # For classifier free guidance, we need to do two forward passes.
    # Here we concatenate the unconditional and text embeddings into a single batch
    # to avoid doing two forward passes

    ################################# modify boom ##############################
    # pdb.set_trace()
    # 3-1. Encoder sub prompt
    list_prompt_embeds = []
    for dot_prompt in layo_prompt:
        text_inputs = self.tokenizer(
            dot_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        dot_prompt_embeds = self.text_encoder(
            text_input_ids.to(device),
        )
        dot_prompt_embeds = dot_prompt_embeds[0]
        list_prompt_embeds.append(dot_prompt_embeds)
    bs_prompt_embeds = torch.stack(list_prompt_embeds).squeeze()  # bs, 77, 768

    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    # 4. Prepare image
    if isinstance(controlnet, ControlNetModel):
        image = self.prepare_image(
            image=image,
            width=width,
            height=height,
            batch_size=batch_size * num_images_per_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            dtype=controlnet.dtype,
            do_classifier_free_guidance=do_classifier_free_guidance,
            guess_mode=guess_mode,
        )
        height, width = image.shape[-2:]
    elif isinstance(controlnet, MultiControlNetModel):
        images = []

        for image_ in image:
            image_ = self.prepare_image(
                image=image_,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guess_mode=guess_mode,
            )

            images.append(image_)

        image = images
        height, width = image[0].shape[-2:]
    else:
        assert False

    # 5. Prepare timesteps
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps

    # 6. Prepare latent variables
    num_channels_latents = self.unet.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 7.1 Create tensor stating which controlnets to keep
    controlnet_keep = []
    for i in range(len(timesteps)):
        keeps = [
            1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
            for s, e in zip(control_guidance_start, control_guidance_end)
        ]
        controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

    def fuse_mask_single_block(f_mask, f_mid_block_res_sample):
        # [4, 8, 8], [4, 1280, 8, 8]
        fus_feat = []
        for mii in range(len(f_mask)):
            mask_block = torch.masked_fill(f_mid_block_res_sample[mii], ~f_mask[mii], 0)
            fus_feat.append(mask_block)
        mask_fus = torch.sum(torch.stack(fus_feat), dim=0)  # [1280, 8, 8]
        return mask_fus

    def fuse_mask_down(f_mask, f_down_block_res_samples):
        #  12, [10, 320, 64, 64] -> 12, [320, 64, 64]
        fus_feat = []
        size_mask = f_mask.shape[-1]
        for ii in range(len(f_down_block_res_samples)):
            dot_down_block_res_samples = f_down_block_res_samples[ii]
            size_dot = dot_down_block_res_samples.shape[-1]
            bins = int(size_mask / size_dot)
            dot_mask = f_mask[:, ::bins, ::bins]
            dot_fuse_block = fuse_mask_single_block(dot_mask, dot_down_block_res_samples)
            fus_feat.append(dot_fuse_block)
        return fus_feat

    all_latents = []
    all_encoders = _to_cpu(prompt_embeds)
    all_cross_attention_kwargs = []
    all_down_block_res_samples = []
    all_mid_block_additional_residual = []

    # pdb.set_trace()
    # 8. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            all_latents.append(_to_cpu(latents))

            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # controlnet(s) inference
            if guess_mode and do_classifier_free_guidance:
                # Infer ControlNet only for the conditional batch.
                control_model_input = latents  # [1, 4, 64, 64]
                control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]

            else:
                control_model_input = latent_model_input  # [2, 4, 64, 64]
                controlnet_prompt_embeds = prompt_embeds  # [2, 77, 768]

            if isinstance(controlnet_keep[i], list):
                cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
            else:
                controlnet_cond_scale = controlnet_conditioning_scale
                if isinstance(controlnet_cond_scale, list):
                    controlnet_cond_scale = controlnet_cond_scale[0]
                cond_scale = controlnet_cond_scale * controlnet_keep[i]

            fuse_down_samples = []
            fuse_mid_samples = []
            if infer_mode == "single":
                # infernce Time Single
                for jj in range(len(image)):
                    dot_prompt_embeds = list_prompt_embeds[jj]
                    dot_prompt_embeds = torch.cat([negative_prompt_embeds, dot_prompt_embeds])
                    controlnet_prompt_embeds = dot_prompt_embeds

                    cond_image = image[jj]
                    down_samples, mid_sample = self.controlnet(
                        control_model_input,  # [2, 4, 64, 64]
                        t,
                        encoder_hidden_states=controlnet_prompt_embeds,  # [2, 77, 768]
                        controlnet_cond=cond_image,  # [2, 3, 512, 512]
                        conditioning_scale=cond_scale,
                        guess_mode=guess_mode,
                        return_dict=False,
                    )

                    if fuse_type == "mask":
                        fuse_down_samples.append(down_samples)
                        fuse_mid_samples.append(mid_sample)

                    if jj == 0:
                        down_block_res_samples, mid_block_res_sample = down_samples, mid_sample
                    else:
                        down_block_res_samples = [
                            samples_prev + samples_curr
                            for samples_prev, samples_curr in zip(down_block_res_samples, down_samples)
                        ]
                        mid_block_res_sample += mid_sample
            elif infer_mode == "batch":
                BNS = len(image)

                dot_prompt_embeds_batch = []
                cond_images_batch = []

                for jj in range(BNS):
                    dot_prompt_embeds = list_prompt_embeds[jj]
                    dot_prompt_embeds = torch.cat([negative_prompt_embeds, dot_prompt_embeds], dim=0)
                    dot_prompt_embeds_batch.append(dot_prompt_embeds)

                    cond_image = image[jj]
                    cond_images_batch.append(cond_image)

                ##########################################################
                dot_prompt_embeds_batch = torch.cat(dot_prompt_embeds_batch, dim=0)  # [4*2, 77, 768]
                cond_images_batch = torch.cat(cond_images_batch, dim=0)  # [4*2, 3, 512, 512]
                control_model_input = torch.repeat_interleave(control_model_input, repeats=BNS,
                                                              dim=0)  # [4*2, 4, 64, 64]

                down_samples, mid_sample = self.controlnet(
                    control_model_input,
                    t,
                    encoder_hidden_states=dot_prompt_embeds_batch,
                    controlnet_cond=cond_images_batch,
                    conditioning_scale=cond_scale,
                    guess_mode=guess_mode,
                    return_dict=False,
                )

                # batch fuse
                down_block_res_samples = [sum(torch.split(sample, 2, dim=0)) for sample in down_samples]
                mid_block_res_sample = sum(torch.split(mid_sample, 2, dim=0))

                if fuse_type == "mask":
                    fuse_down_samples = [torch.split(sample, 2, dim=0) for sample in down_samples]
                    fuse_mid_samples = torch.split(mid_sample, 2, dim=0)

            if fuse_type == "avg":
                mid_block_res_sample = mid_block_res_sample / len(image)  # [2, 1280, 8, 8]
                down_block_res_samples = [d / len(image) for d in down_block_res_samples]  # 12, [[2, 320, 64, 64], ...]
            elif fuse_type == "mask":
                # mask fuse
                pass
            else:
                pass

            if guess_mode and do_classifier_free_guidance:
                # Infered ControlNet only for the conditional batch.
                # To apply the output of ControlNet to both the unconditional and conditional batches,
                # add 0 to the unconditional batch to keep it unchanged.
                down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                return_dict=False,
            )[0]

            all_cross_attention_kwargs.append(_to_cpu(cross_attention_kwargs) if cross_attention_kwargs is not None else None)
            all_down_block_res_samples.append(_to_cpu(down_block_res_samples))    
            all_mid_block_additional_residual.append(_to_cpu(mid_block_res_sample)) 

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents, log_prob= dpm_step_with_logprob(
                self.scheduler, noise_pred, t, latents, s_churn=s_churn
            )

            # torch.save(latents, "dm_latents_%s.pt" % i)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

    all_latents.append(latents)
    # If we do sequential model offloading, let's offload unet and controlnet
    # manually for max memory savings
    if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
        self.unet.to("cpu")
        self.controlnet.to("cpu")
        torch.cuda.empty_cache()

    if not output_type == "latent":
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
    else:
        image = latents
        has_nsfw_concept = None

    if has_nsfw_concept is None:
        do_denormalize = [True] * image.shape[0]
    else:
        do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

    image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

    # Offload last model to CPU
    if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
        self.final_offload_hook.offload()

    return image, all_latents, all_encoders, all_cross_attention_kwargs, all_down_block_res_samples, all_mid_block_additional_residual, timesteps