import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import comfy.conds
import folder_paths

if "comfyui_iris_tests" not in sys.modules:
    pack = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location(
        "comfyui_iris_tests", pack / "__init__.py", submodule_search_locations=[str(pack)]
    )
    iris = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = iris
    spec.loader.exec_module(iris)

from comfyui_iris_tests.nodes import IrisCLIPLoader, IrisKSampler, IrisModelLoader, iris_sigmas
from comfyui_iris_tests.integration import IrisModel


def test_loaders_and_sampler_expose_standard_comfy_types():
    assert IrisModelLoader.define_schema().outputs[0].io_type == "MODEL"
    assert IrisCLIPLoader.define_schema().outputs[0].io_type == "CLIP"
    sampler = IrisKSampler.define_schema()
    assert sampler.outputs[0].io_type == "IMAGE"
    assert not any(spec.io_type == "VAE" for spec in sampler.inputs)
    scheduler = next(spec for spec in sampler.inputs if spec.id == "scheduler")
    assert scheduler.default == "iris"


def test_model_extra_conds_returns_standard_cross_attention_conditioning():
    context = torch.ones(1, 2, 4)
    model = SimpleNamespace(
        diffusion_model=SimpleNamespace(preprocess_text=lambda states, mask: states + mask[..., None]),
        get_dtype_inference=lambda: torch.float32,
    )

    result = IrisModel.extra_conds(
        model,
        device=torch.device("cpu"),
        cross_attn=context,
        attention_mask=torch.tensor([[1, 0]]),
    )

    assert set(result) == {"c_crossattn"}
    assert isinstance(result["c_crossattn"], comfy.conds.CONDRegular)
    torch.testing.assert_close(
        result["c_crossattn"].cond,
        torch.tensor([[[2, 2, 2, 2], [1, 1, 1, 1]]], dtype=torch.float32),
    )


@pytest.mark.parametrize(
    ("loader", "name_input"),
    [(IrisModelLoader, "model_name"), (IrisCLIPLoader, "clip_name")],
)
def test_loaders_resolve_untrusted_names_inside_configured_model_folders(loader, name_input, monkeypatch, tmp_path):
    monkeypatch.setattr(
        folder_paths,
        "folder_names_and_paths",
        {
            "diffusion_models": ([str(tmp_path)], set()),
            "text_encoders": ([str(tmp_path)], set()),
        },
    )
    inputs = {name_input: "../../../../etc/passwd"}
    if loader is IrisModelLoader:
        inputs["weight_dtype"] = "default"
    with pytest.raises(FileNotFoundError):
        loader.execute(**inputs)


def test_sampler_returns_normalized_nhwc_image_and_forwards_seed(monkeypatch):
    from comfyui_iris_tests import nodes

    monkeypatch.setattr(nodes.comfy.model_management, "intermediate_device", lambda: torch.device("cpu"))
    call = {}

    class PixelModel:
        def get_model_object(self, key):
            assert key == "latent_format"
            return SimpleNamespace(latent_channels=3, spacial_downscale_ratio=1, latent_dimensions=2)

    def fake_ksampler(*args):
        call["args"] = args
        samples = torch.tensor([[[[-1.0, 0.0]], [[0.5, 1.0]], [[1.0, -0.5]]]])
        return ({"samples": samples},)

    monkeypatch.setattr(nodes.nodes, "common_ksampler", fake_ksampler)
    result = IrisKSampler.execute(
        model=PixelModel(), positive=[], negative=[], seed=17, steps=2, cfg=1.5,
        sampler_name="euler", scheduler="simple", width=2, height=1, batch_size=1,
    )

    assert call["args"][1] == 17
    assert call["args"][8]["samples"].shape == (1, 3, 1, 2)
    assert result.result[0].shape == (1, 1, 2, 3)
    torch.testing.assert_close(
        result.result[0],
        torch.tensor([[[[0.0, 0.75, 1.0], [0.5, 1.0, 0.25]]]]),
    )


def test_sampler_rejects_non_pixel_model():
    model = SimpleNamespace(
        get_model_object=lambda key: SimpleNamespace(
            latent_channels=4, spacial_downscale_ratio=8, latent_dimensions=2
        )
    )

    with pytest.raises(ValueError, match="pixel-space RGB"):
        IrisKSampler.execute(
            model=model, positive=[], negative=[], seed=0, steps=1, cfg=1.0,
            sampler_name="euler", scheduler="simple", width=2, height=2, batch_size=1,
        )


def test_iris_flow_sigmas_follow_shifted_upstream_schedule():
    sigmas = iris_sigmas(4)

    assert sigmas.shape == (5,)
    assert sigmas.dtype == torch.float32
    torch.testing.assert_close(sigmas[0], torch.tensor(4 * 0.999 / (1 + 3 * 0.999)))
    assert sigmas[-1].item() == 0.0
    assert torch.all(sigmas[:-1] >= sigmas[1:])


def test_iris_sampler_passes_its_grid_and_seeded_noise_to_comfy_sampler(monkeypatch):
    from comfyui_iris_tests import nodes

    monkeypatch.setattr(nodes.comfy.model_management, "intermediate_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(nodes.latent_preview, "prepare_callback", lambda model, steps: None)
    calls = []

    class PixelModel:
        def get_model_object(self, key):
            return SimpleNamespace(latent_channels=3, spacial_downscale_ratio=1, latent_dimensions=2)

    def fake_sample(*args, **kwargs):
        calls.append((args, kwargs))
        return torch.zeros_like(args[8])

    monkeypatch.setattr(nodes.comfy.sample, "sample", fake_sample)
    outputs = [
        IrisKSampler.execute(
            model=PixelModel(), positive=[], negative=[], seed=23, steps=3, cfg=1.0,
            sampler_name="euler", scheduler="iris", width=2, height=2, batch_size=1,
        )
        for _ in range(2)
    ]

    (first_args, first_kwargs), (second_args, second_kwargs) = calls
    assert first_args[2:6] == (3, 1.0, "euler", "simple")
    assert first_kwargs["seed"] == 23
    torch.testing.assert_close(first_kwargs["sigmas"], iris_sigmas(3))
    torch.testing.assert_close(first_args[1], second_args[1], rtol=0, atol=0)
    torch.testing.assert_close(first_kwargs["sigmas"], second_kwargs["sigmas"], rtol=0, atol=0)
    assert outputs[0].result[0].shape == (1, 2, 2, 3)
