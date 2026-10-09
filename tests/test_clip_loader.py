import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

if "comfyui_iris_tests" not in sys.modules:
    pack = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location(
        "comfyui_iris_tests", pack / "__init__.py", submodule_search_locations=[str(pack)]
    )
    iris = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = iris
    spec.loader.exec_module(iris)

from comfyui_iris_tests import integration, text_encoder


def language_state(prefix="model.language_model."):
    layer_weights = (
        "input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm",
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    )
    state = {
        prefix + "embed_tokens.weight": torch.ones(1),
        prefix + "norm.weight": torch.ones(2560),
    }
    for layer in range(36):
        for name in layer_weights:
            size = 1
            if layer == 0 and name == "self_attn.q_norm":
                size = 128
            elif layer == 0 and name == "self_attn.k_proj":
                size = 1024
            elif layer == 35 and name == "self_attn.q_proj":
                size = 4096
            state[f"{prefix}layers.{layer}.{name}.weight"] = torch.ones(size)
    return state


class CapturingClip:
    def __init__(self, target, embedding_directory, parameters, state_dict, model_options):
        self.target = target
        self.state_dict = state_dict[0]
        self.model_options = model_options


@pytest.mark.parametrize("prefix", ["model.language_model.", "language_model.", "model."])
def test_clip_loader_normalizes_compatible_qwen_namespaces(monkeypatch, prefix):
    source = language_state(prefix)
    monkeypatch.setattr(integration.comfy.utils, "load_torch_file", lambda *a, **k: (source, {}))
    monkeypatch.setattr(integration.comfy.sd, "CLIP", CapturingClip)

    clip = integration.load_clip("fine-tuned-qwen.safetensors")

    assert len(clip.state_dict) == 398
    assert "model.embed_tokens.weight" in clip.state_dict
    assert "model.layers.35.self_attn.q_proj.weight" in clip.state_dict
    assert all(
        not key.startswith("model.language_model.") and not key.startswith("language_model.")
        for key in clip.state_dict
    )
    assert clip.target.clip is text_encoder.IrisTE


def test_clip_loader_passes_converted_quant_metadata_to_native_text_model(monkeypatch):
    source = language_state()
    metadata = {
        "_quantization_metadata": json.dumps(
            {"layers": {"model.language_model.layers.0.self_attn.q_proj": {"format": "int8"}}}
        )
    }
    monkeypatch.setattr(integration.comfy.utils, "load_torch_file", lambda *a, **k: (source, metadata))
    monkeypatch.setattr(integration.comfy.sd, "CLIP", CapturingClip)
    captured = {}

    def initialize_sd1_clip(self, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(text_encoder.sd1_clip.SD1ClipModel, "__init__", initialize_sd1_clip)
    clip = integration.load_clip("qwen-int8.safetensors")
    clip.target.clip(model_options=clip.model_options)

    expected = {"quantization_metadata": {"mixed_ops": True}}
    assert clip.model_options == expected
    assert captured["model_options"] == expected
    assert "model.layers.0.self_attn.q_proj.comfy_quant" in clip.state_dict


def test_qwen_clip_forwards_quant_options_to_comfy_native_model(monkeypatch):
    captured = {}

    def initialize_sd_clip(self, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(text_encoder.sd1_clip.SDClipModel, "__init__", initialize_sd_clip)
    quant = {"quantization_metadata": {"mixed_ops": True}}

    text_encoder.IrisQwenClip(model_options=quant)

    assert captured["model_options"] == quant
    assert captured["model_class"] is text_encoder.IrisLanguageModel


def test_clip_loader_rejects_other_or_incomplete_language_architectures(monkeypatch):
    source = language_state()
    source["model.language_model.layers.0.self_attn.q_norm.weight"] = torch.ones(64)
    monkeypatch.setattr(integration.comfy.utils, "load_torch_file", lambda *a, **k: (source, {}))
    monkeypatch.setattr(
        integration.comfy.sd,
        "CLIP",
        lambda **kwargs: pytest.fail("unsupported architecture reached CLIP construction"),
    )

    with pytest.raises(ValueError, match="Qwen3-VL-4B's 36-layer, 2560-wide language architecture"):
        integration.load_clip("wrong-architecture.safetensors")


def test_clip_loader_rejects_quantized_layer_without_its_weight(monkeypatch):
    source = language_state()
    source.pop("model.language_model.layers.4.self_attn.q_proj.weight")
    metadata = {
        "_quantization_metadata": json.dumps(
            {"layers": {"model.language_model.layers.4.self_attn.q_proj": {"format": "int8"}}}
        )
    }
    monkeypatch.setattr(integration.comfy.utils, "load_torch_file", lambda *a, **k: (source, metadata))
    monkeypatch.setattr(
        integration.comfy.sd,
        "CLIP",
        lambda **kwargs: pytest.fail("incomplete quantized weights reached CLIP construction"),
    )

    with pytest.raises(ValueError, match="Incomplete Qwen3-VL-4B language weights: missing=.*layers.4.self_attn.q_proj.weight"):
        integration.load_clip("missing-quantized-weight.safetensors")
