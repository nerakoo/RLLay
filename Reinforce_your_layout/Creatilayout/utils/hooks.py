# hooks.py
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from diffusers.utils import convert_unet_state_dict_to_peft

def unwrap_model(model, accelerator, is_compiled_module):
    model = accelerator.unwrap_model(model)
    return model._orig_mod if is_compiled_module(model) else model

def save_model_hook(models, weights, output_dir,
                    accelerator, transformer_type, get_peft_state_fn, pipeline):
    if not accelerator.is_main_process:
        return

    lora_dict = None
    for model in models:
        # 这里用 isinstance 判断 transformer
        if isinstance(model, transformer_type):
            lora_dict = get_peft_state_fn(model)
        else:
            raise ValueError(f"unexpected model: {type(model)}")
        weights.pop()

    pipeline.save_lora_weights(
        output_dir,
        transformer_lora_layers=lora_dict,
        text_encoder_lora_layers=None,
    )

def load_model_hook(models, input_dir,
                    accelerator, transformer_type, set_peft_state_fn,
                    pipeline_cls, convert_fn, cast_fn, config, logger):
    transformer_ = None
    while models:
        m = models.pop()
        if isinstance(m, transformer_type):
            transformer_ = m
        else:
            raise ValueError(f"unexpected model: {type(m)}")

    lora_state = pipeline_cls.lora_state_dict(input_dir)
    # 只保留 transformer 的那部分
    ts = {k.replace("transformer.", ""): v
          for k, v in lora_state.items() if k.startswith("transformer.")}
    ts = convert_fn(ts)
    incomp = set_peft_state_fn(transformer_, ts, adapter_name="default")
    if incomp and getattr(incomp, "unexpected_keys", None):
        logger.warning(f"got unexpected keys: {incomp.unexpected_keys}")

    if config.mixed_precision == "fp16":
        cast_fn([transformer_])
