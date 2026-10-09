import torch

import comfy.model_management
import comfy.samplers
import comfy.sample
import comfy.utils
import latent_preview
import folder_paths
import nodes
from comfy_api.latest import io

from .integration import load_model, load_clip


class IrisModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="IrisModelLoader", display_name="Load Iris Model", category="loaders/iris", inputs=[io.Combo.Input("model_name", options=folder_paths.get_filename_list("diffusion_models")), io.Combo.Input("weight_dtype", options=["default", "bfloat16", "float32"])], outputs=[io.Model.Output()])

    @classmethod
    def execute(cls, model_name, weight_dtype):
        path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        return io.NodeOutput(load_model(path, weight_dtype))


class IrisCLIPLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="IrisCLIPLoader", display_name="Load Iris CLIP", category="loaders/iris", inputs=[io.Combo.Input("clip_name", options=folder_paths.get_filename_list("text_encoders"))], outputs=[io.Clip.Output()])

    @classmethod
    def execute(cls, clip_name):
        path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
        return io.NodeOutput(load_clip(path))


class IrisKSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="IrisKSampler", display_name="Iris KSampler (Pixels)", category="sampling/iris", description="Sample RGB pixels directly using ComfyUI samplers. No VAE required.", inputs=[
            io.Model.Input("model"), io.Conditioning.Input("positive"), io.Conditioning.Input("negative"),
            io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
            io.Int.Input("steps", default=100, min=1, max=10000), io.Float.Input("cfg", default=3.0, min=0.0, max=100.0, step=0.1),
            io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS, default="dpmpp_2m"),
            io.Combo.Input("scheduler", options=["iris"] + comfy.samplers.KSampler.SCHEDULERS, default="iris"),
            io.Int.Input("width", default=1024, min=1, max=nodes.MAX_RESOLUTION), io.Int.Input("height", default=1024, min=1, max=nodes.MAX_RESOLUTION),
            io.Int.Input("batch_size", default=1, min=1, max=4096),
        ], outputs=[io.Image.Output()])

    @classmethod
    def execute(cls, model, positive, negative, seed, steps, cfg, sampler_name, scheduler, width, height, batch_size):
        latent_format = model.get_model_object("latent_format")
        if latent_format.latent_channels != 3 or latent_format.spacial_downscale_ratio != 1 or latent_format.latent_dimensions != 2:
            raise ValueError("Iris KSampler requires a pixel-space RGB model.")
        latent = {"samples": torch.zeros((batch_size, 3, height, width), device=comfy.model_management.intermediate_device())}
        if scheduler == "iris":
            samples = latent["samples"]
            noise = comfy.sample.prepare_noise(samples, seed)
            sigmas = iris_sigmas(steps)
            sampled = {"samples": comfy.sample.sample(model, noise, steps, cfg, sampler_name, "simple", positive, negative, samples, sigmas=sigmas, callback=latent_preview.prepare_callback(model, steps), disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed)}
        else:
            sampled = nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent)[0]
        return io.NodeOutput((sampled["samples"].movedim(1, -1) + 1.0) / 2.0)


def iris_sigmas(steps):
    sigma = 1.0 - torch.linspace(1.0, 0.001, steps + 1, dtype=torch.float64)
    return (4.0 * sigma / (1.0 + 3.0 * sigma)).flip(0).to(torch.float32)
