"""Local checkpoint loading and the standard ComfyUI MODEL contract."""
import torch

import comfy.conds
import comfy.latent_formats
import comfy.model_base
import comfy.model_management
import comfy.model_patcher
import comfy.sd
import comfy.storage
import comfy.supported_models_base
import comfy.utils

from .model import IrisConfig, IrisDiT
from .text_encoder import IrisTokenizer, IrisTE


class IrisModelConfig(comfy.supported_models_base.BASE):
    unet_extra_config = {}
    latent_format = comfy.latent_formats.ChromaRadiance
    sampling_settings = {"shift": 4.0, "multiplier": 1000}
    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]
    memory_usage_factor = 0.3


class IrisModel(comfy.model_base.BaseModel):
    def __init__(self, config, device=None):
        super().__init__(config, comfy.model_base.ModelType.FLOW, device=device, unet_model=IrisDiT)

    def extra_conds(self, **kwargs):
        device = kwargs["device"]
        dtype = self.get_dtype_inference()
        states = kwargs["cross_attn"].to(device=device, dtype=dtype)
        mask = kwargs["attention_mask"].to(device=device)
        context = self.diffusion_model.preprocess_text(states, mask)
        return {"c_crossattn": comfy.conds.CONDRegular(context)}


def load_model(path, weight_dtype="default"):
    sd = comfy.utils.load_torch_file(path)
    config = IrisModelConfig({"cfg": IrisConfig()})
    config.quant_config = comfy.utils.detect_layer_quantization(sd, "")
    load_device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()
    parameters = comfy.utils.calculate_parameters(sd)
    stored_dtype = None if config.quant_config else comfy.utils.weight_dtype(sd)
    dtype = {"default": None, "bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[weight_dtype]
    if dtype is None:
        dtype = comfy.model_management.unet_dtype(model_params=parameters, supported_dtypes=config.supported_inference_dtypes, weight_dtype=stored_dtype)
    manual_cast = comfy.model_management.unet_manual_cast(None if config.quant_config else dtype, load_device, config.supported_inference_dtypes)
    config.set_inference_dtype(dtype, manual_cast, device=load_device)
    model = IrisModel(config, device=offload_device)
    patcher = comfy.model_patcher.CoreModelPatcher(model, load_device=load_device, offload_device=offload_device, fast_disk=comfy.storage.state_dict_fast_disk(sd))
    model.diffusion_model.load_state_dict(sd, strict=True, assign=patcher.is_dynamic())
    return patcher


def load_clip(path):
    sd, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
    sd, _ = comfy.utils.convert_old_quants(sd, model_prefix="", metadata=metadata)
    prefixes = ("model.language_model.", "language_model.", "model.")
    prefix = next((prefix for prefix in prefixes if prefix + "norm.weight" in sd), None)
    if prefix is None:
        raise ValueError("Iris requires compatible Qwen3-VL-4B language weights.")
    text_sd = comfy.utils.state_dict_prefix_replace(sd, {prefix: "model."}, filter_keys=True)
    signature = {"model.norm.weight": 2560,
                 "model.layers.0.self_attn.q_norm.weight": 128,
                 "model.layers.0.self_attn.k_proj.weight": 1024,
                 "model.layers.35.self_attn.q_proj.weight": 4096}
    if any(key not in text_sd or text_sd[key].shape[0] != size for key, size in signature.items()) or any(key.startswith("model.layers.36.") for key in text_sd):
        raise ValueError("Iris requires Qwen3-VL-4B's 36-layer, 2560-wide language architecture.")
    layer_weights = ("input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
    required = ["model.embed_tokens.weight", "model.norm.weight"]
    required += [f"model.layers.{layer}.{name}.weight" for layer in range(36) for name in layer_weights]
    missing = [key for key in required if key not in text_sd]
    if missing:
        raise ValueError(f"Incomplete Qwen3-VL-4B language weights: missing={missing}")
    quant = comfy.utils.detect_layer_quantization(text_sd, "")
    model_options = {"quantization_metadata": quant} if quant is not None else {}
    target = comfy.supported_models_base.ClipTarget(IrisTokenizer, IrisTE)
    return comfy.sd.CLIP(target, embedding_directory=None, parameters=comfy.utils.calculate_parameters(text_sd), state_dict=[text_sd], model_options=model_options)
