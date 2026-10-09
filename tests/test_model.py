import importlib.util
from pathlib import Path
import sys

import pytest
import torch

model_path = Path(__file__).parents[1] / "model.py"
spec = importlib.util.spec_from_file_location("iris_native_model", model_path)
iris_model = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = iris_model
spec.loader.exec_module(iris_model)
IrisConfig, IrisDiT = iris_model.IrisConfig, iris_model.IrisDiT


def tiny_model():
    cfg = IrisConfig(
        hidden_size=8,
        depth=2,
        dual_depth=1,
        num_heads=2,
        num_kv_heads=1,
        patch_size=2,
        in_channels=3,
        mlp_ratio=1.5,
        text_dim=8,
        text_len=3,
        text_lap_num_layers=2,
        text_lap_num_heads=2,
        text_lap_mlp_ratio=1.5,
        pixel_hidden_size=4,
        pixel_attn_hidden_size=8,
        pixel_num_heads=2,
        pixel_depth=1,
        pixel_mlp_ratio=2.0,
    )
    model = IrisDiT(cfg, device="cpu", dtype=torch.float32)
    for index, parameter in enumerate(model.parameters()):
        parameter.data.fill_((index % 7 + 1) * 0.01)
    return model


def test_tiny_model_preprocesses_layerwise_text_and_crops_padded_pixels():
    model = tiny_model()
    state = model.state_dict()
    assert state["blocks.0.attn.k_proj_x.weight"].shape == (4, 8)
    assert state["blocks.1.k_proj.weight"].shape == (4, 8)
    assert state["y_embedder.layer_pool.weight"].shape == (1, 2)
    assert state["pixel_blocks.0.adaln.weight"].shape == (64, 8)

    context = torch.arange(1 * 3 * 2 * 8, dtype=torch.float32).reshape(1, 3, 2, 8) / 100
    mask = torch.tensor([[1, 1, 0]])

    prepared = model.preprocess_text(context, mask)
    image = torch.linspace(-1, 1, 1 * 3 * 5 * 7).reshape(1, 3, 5, 7)
    result = model(image, torch.tensor([250.0]), prepared)

    assert prepared.shape == (1, 3, 8)
    assert result.shape == image.shape
    assert torch.isfinite(result).all()


def test_tiny_model_forward_is_repeatable_for_same_inputs():
    model = tiny_model()
    context = torch.ones(1, 2, 2, 8)
    mask = torch.tensor([[1, 1]])
    prepared = model.preprocess_text(context, mask)
    image = torch.zeros(1, 3, 4, 6)
    timestep = torch.tensor([500.0])

    first = model(image, timestep, prepared)
    second = model(image, timestep, prepared)

    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_tiny_model_matches_upstream_outputs_and_checkpoint_keys():
    upstream = Path(__file__).parents[3] / "tmp/iris3b/upstream/src"
    if not (upstream / "iris3b").is_dir():
        pytest.skip("the task-local Iris upstream checkout is unavailable")
    pytest.importorskip("omegaconf")
    sys.path.insert(0, str(upstream))

    from iris3b.config import ModelConfig, PixelStageConfig
    from iris3b.models.dit import IrisDiT as UpstreamIrisDiT

    native = tiny_model()
    cfg = native.cfg
    upstream_cfg = ModelConfig(
        block="single_stream",
        dual_depth=cfg.dual_depth,
        hidden_size=cfg.hidden_size,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        num_kv_heads=cfg.num_kv_heads,
        patch_size=cfg.patch_size,
        in_channels=cfg.in_channels,
        mlp_ratio=cfg.mlp_ratio,
        norm_eps=cfg.norm_eps,
        text_dim=cfg.text_dim,
        text_len=cfg.text_len,
        text_lap_num_layers=cfg.text_lap_num_layers,
        text_lap_num_heads=cfg.text_lap_num_heads,
        text_lap_mlp_ratio=cfg.text_lap_mlp_ratio,
        pixel=PixelStageConfig(
            depth=cfg.pixel_depth,
            hidden_size=cfg.pixel_hidden_size,
            attn_hidden_size=cfg.pixel_attn_hidden_size,
            num_heads=cfg.pixel_num_heads,
            mlp_ratio=cfg.pixel_mlp_ratio,
            modulation="post",
        ),
    )
    reference = UpstreamIrisDiT(upstream_cfg).eval()
    native.load_state_dict(reference.state_dict(), strict=True)

    image = torch.linspace(-1, 1, 1 * 3 * 4 * 6).reshape(1, 3, 4, 6)
    text = torch.linspace(-0.5, 0.5, 1 * 3 * 2 * 8).reshape(1, 3, 2, 8)
    mask = torch.tensor([[1, 1, 0]])
    timestep = torch.tensor([250.0])
    expected = reference(image, timestep, text, y_mask=mask).x
    actual = native(image, timestep, native.preprocess_text(text, mask))

    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
