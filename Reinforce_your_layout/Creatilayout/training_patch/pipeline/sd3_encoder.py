from typing import List, Optional, Union
import torch

def _module_device(module, fallback: Optional[torch.device] = None) -> torch.device:
    if fallback is not None:
        return fallback
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")

def _module_dtype(module) -> torch.dtype:
    return getattr(module, "dtype", next(module.parameters()).dtype)

def encode_prompt_with_t5(
    text_encoder,             
    tokenizer,                
    prompt: Union[str, List[str]],
    max_sequence_length: int = 256,
    num_images_per_prompt: int = 1,
    device: Optional[torch.device] = None,
    text_input_ids: Optional[torch.LongTensor] = None,
) -> torch.FloatTensor:
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        if text_input_ids is not None:
            raise ValueError("Pass either `tokenizer` or `text_input_ids`, not both.")
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("`text_input_ids` must be provided when `tokenizer` is None`.")

    dev = _module_device(text_encoder, device)
    out = text_encoder(text_input_ids.to(dev))[0]   
    out = out.to(dtype=_module_dtype(text_encoder), device=dev)

    _, seq_len, _ = out.shape
    out = out.repeat(1, num_images_per_prompt, 1)
    out = out.view(batch_size * num_images_per_prompt, seq_len, -1)
    return out

def encode_prompt_with_clip(
    text_encoder,             
    tokenizer,                 
    prompt: Union[str, List[str]],
    num_images_per_prompt: int = 1,
    device: Optional[torch.device] = None,
    text_input_ids: Optional[torch.LongTensor] = None,
    clip_skip: Optional[int] = None,
    max_length: Optional[int] = None, 
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        if text_input_ids is not None:
            raise ValueError("Pass either `tokenizer` or `text_input_ids`, not both.")
        if max_length is None:
            max_length = getattr(tokenizer, "model_max_length", 77)
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("`text_input_ids` must be provided when `tokenizer` is None`.")

    dev = _module_device(text_encoder, device)
    outputs = text_encoder(text_input_ids.to(dev), output_hidden_states=True)

    pooled = outputs[0]  
    hs = outputs.hidden_states

    take = -2 if clip_skip is None else -(clip_skip + 2)
    hidden = hs[take].to(dtype=_module_dtype(text_encoder), device=dev)

    _, seq_len, _ = hidden.shape
    hidden = hidden.repeat(1, num_images_per_prompt, 1)
    hidden = hidden.view(batch_size * num_images_per_prompt, seq_len, -1)

    return hidden, pooled

def encode_prompt(
    text_encoders: List,    
    tokenizers: List,       
    prompt: Union[str, List[str]],
    max_sequence_length: int = 256,
    device: Optional[torch.device] = None,
    num_images_per_prompt: int = 1,
    text_input_ids_list: Optional[List[Optional[torch.LongTensor]]] = None, 
    clip_skip: Optional[int] = None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    assert len(text_encoders) >= 3 and len(tokenizers) >= 3, 

    clip_text_encoders = text_encoders[:2]
    clip_tokenizers = tokenizers[:2]

    ids_list = text_input_ids_list if text_input_ids_list is not None else [None, None, None]

    clip_hidden_list = []
    clip_pooled_list = []
    for i, (enc, tok) in enumerate(zip(clip_text_encoders, clip_tokenizers)):
        hidden_i, pooled_i = encode_prompt_with_clip(
            text_encoder=enc,
            tokenizer=tok,
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            text_input_ids=ids_list[i],
            clip_skip=clip_skip,
        )
        clip_hidden_list.append(hidden_i)
        clip_pooled_list.append(pooled_i)

    clip_prompt_embeds = torch.cat(clip_hidden_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_list, dim=-1)

    t5_enc = text_encoders[-1]
    t5_tok = tokenizers[-1]
    t5_prompt_embeds = encode_prompt_with_t5(
        text_encoder=t5_enc,
        tokenizer=t5_tok,
        prompt=prompt,
        max_sequence_length=max_sequence_length,
        num_images_per_prompt=num_images_per_prompt,
        device=device,
        text_input_ids=ids_list[-1],
    )

    pad_needed = t5_prompt_embeds.shape[-1] - clip_prompt_embeds.shape[-1]
    if pad_needed > 0:
        clip_prompt_embeds = torch.nn.functional.pad(clip_prompt_embeds, (0, pad_needed))
    elif pad_needed < 0:
        clip_prompt_embeds = clip_prompt_embeds[..., : t5_prompt_embeds.shape[-1]]

    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embeds], dim=-2)

    return prompt_embeds, pooled_prompt_embeds
