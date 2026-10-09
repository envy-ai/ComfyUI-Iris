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
    inputs = {spec.id: spec for spec in sampler.inputs}
    assert inputs["image"].io_type == "IMAGE"
    assert inputs["image"].optional
    assert inputs["denoise"].optional
    assert inputs["denoise"].default == 1.0


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

    def fake_ksampler(*args, **kwargs):
        call["args"] = args
        call["kwargs"] = kwargs
        samples = torch.empty((1, 3, 64, 64))
        samples[:, 0].fill_(-1.0)
        samples[:, 1].fill_(0.5)
        samples[:, 2].fill_(1.0)
        return ({"samples": samples},)

    monkeypatch.setattr(nodes.nodes, "common_ksampler", fake_ksampler)
    result = IrisKSampler.execute(
        model=PixelModel(), positive=[], negative=[], seed=17, steps=2, cfg=1.5,
        sampler_name="euler", scheduler="simple", width=65, height=67, batch_size=1,
    )

    assert call["args"][1] == 17
    assert call["args"][8]["samples"].shape == (1, 3, 64, 64)
    assert call["kwargs"]["denoise"] == 1.0
    assert result.result[0].shape == (1, 64, 64, 3)
    torch.testing.assert_close(result.result[0][0, 0, 0], torch.tensor([0.0, 0.75, 1.0]))


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
            sampler_name="euler", scheduler="iris", width=65, height=67, batch_size=1,
        )
        for _ in range(2)
    ]

    (first_args, first_kwargs), (second_args, second_kwargs) = calls
    assert first_args[2:6] == (3, 1.0, "euler", "simple")
    assert first_kwargs["seed"] == 23
    assert first_kwargs["denoise"] == 1.0
    torch.testing.assert_close(first_kwargs["sigmas"], iris_sigmas(3))
    torch.testing.assert_close(first_args[1], second_args[1], rtol=0, atol=0)
    torch.testing.assert_close(first_kwargs["sigmas"], second_kwargs["sigmas"], rtol=0, atol=0)
    assert outputs[0].result[0].shape == (1, 64, 64, 3)


def test_sampler_image_overrides_widget_size_batch_and_maps_rgb_to_pixel_range(monkeypatch):
    from comfyui_iris_tests import nodes

    monkeypatch.setattr(nodes.comfy.model_management, "intermediate_device", lambda: torch.device("cpu"))
    image = torch.empty((2, 37, 53, 3), dtype=torch.float32)
    image[..., 0] = 0.0
    image[..., 1] = 0.25
    image[..., 2] = 1.0
    captured = {}

    class PixelModel:
        def get_model_object(self, key):
            assert key == "latent_format"
            return SimpleNamespace(latent_channels=3, spacial_downscale_ratio=1, latent_dimensions=2)

    def fake_ksampler(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return ({"samples": args[8]["samples"]},)

    monkeypatch.setattr(nodes.nodes, "common_ksampler", fake_ksampler)
    result = IrisKSampler.execute(
        model=PixelModel(), positive=[], negative=[], seed=17, steps=4, cfg=1.5,
        sampler_name="euler", scheduler="simple", width=11, height=13, batch_size=7,
        image=image, denoise=0.4,
    )

    latent = captured["args"][8]["samples"]
    assert latent.shape == (2, 3, 32, 48)
    assert captured["kwargs"]["denoise"] == 0.4
    expected_latent = torch.empty((2, 3, 32, 48))
    expected_latent[:, 0].fill_(-1.0)
    expected_latent[:, 1].fill_(-0.5)
    expected_latent[:, 2].fill_(1.0)
    torch.testing.assert_close(latent, expected_latent, rtol=0, atol=2e-7)
    assert result.result[0].shape == (2, 32, 48, 3)
    torch.testing.assert_close(result.result[0][0, 0, 0], torch.tensor([0.0, 0.25, 1.0]))


def test_sampler_zero_denoise_returns_connected_image_without_sampling(monkeypatch):
    from comfyui_iris_tests import nodes

    image = torch.rand((2, 32, 48, 3))
    monkeypatch.setattr(
        nodes.comfy.sample,
        "sample",
        lambda *args, **kwargs: pytest.fail("denoise=0 must bypass sampling"),
    )
    monkeypatch.setattr(
        nodes.nodes,
        "common_ksampler",
        lambda *args, **kwargs: pytest.fail("denoise=0 must bypass sampling"),
    )

    result = IrisKSampler.execute(
        model=object(), positive=[], negative=[], seed=2, steps=4, cfg=1.0,
        sampler_name="euler", scheduler="iris", width=10, height=10, batch_size=1,
        image=image, denoise=0.0,
    )

    assert result.result[0] is image


def test_sampler_zero_denoise_resizes_odd_image_to_output_grid():
    image = torch.empty((2, 37, 53, 3))
    image[..., 0] = 0.0
    image[..., 1] = 0.25
    image[..., 2] = 1.0

    result = IrisKSampler.execute(
        model=object(), positive=[], negative=[], seed=2, steps=4, cfg=1.0,
        sampler_name="euler", scheduler="iris", width=11, height=13, batch_size=1,
        image=image, denoise=0.0,
    )

    assert result.result[0].shape == (2, 32, 48, 3)
    assert result.result[0] is not image
    torch.testing.assert_close(result.result[0][0, 0, 0], torch.tensor([0.0, 0.25, 1.0]))


def test_iris_partial_denoise_uses_native_trailing_sigma_grid(monkeypatch):
    from comfyui_iris_tests import nodes

    monkeypatch.setattr(nodes.comfy.model_management, "intermediate_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(nodes.latent_preview, "prepare_callback", lambda model, steps: None)
    image = torch.full((2, 37, 53, 3), 0.25)
    captured = {}

    class PixelModel:
        def get_model_object(self, key):
            assert key == "latent_format"
            return SimpleNamespace(latent_channels=3, spacial_downscale_ratio=1, latent_dimensions=2)

    def fake_sample(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return args[8]

    monkeypatch.setattr(nodes.comfy.sample, "sample", fake_sample)
    IrisKSampler.execute(
        model=PixelModel(), positive=[], negative=[], seed=31, steps=4, cfg=2.0,
        sampler_name="euler", scheduler="iris", width=9, height=9, batch_size=1,
        image=image, denoise=0.5,
    )

    assert captured["kwargs"]["denoise"] == 0.5
    torch.testing.assert_close(captured["kwargs"]["sigmas"], iris_sigmas(8)[-5:])
    assert captured["args"][8].shape == (2, 3, 32, 48)
