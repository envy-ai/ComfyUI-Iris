import torch

import comfy.model_management
import comfy.samplers
import comfy.sample
import comfy.utils
import latent_preview
import folder_paths
import nodes
from comfy_api.latest import io
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced

from .integration import load_model, load_clip


class IrisModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="IrisModelLoader", display_name="Load Iris Model", category="loaders/iris", inputs=[io.Combo.Input("model_name", options=folder_paths.get_filename_list("diffusion_models")), io.Combo.Input("weight_dtype", options=["default", "bfloat16", "float16", "float32"])], outputs=[io.Model.Output()])

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
            io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01, optional=True),
            io.Image.Input("image", optional=True, tooltip="Sets dimensions and batch size; dimensions round to the nearest 16 pixels. Denoise controls image-to-image strength."),
        ], outputs=[io.Image.Output()])

    @classmethod
    def execute(cls, model, positive, negative, seed, steps, cfg, sampler_name, scheduler, width, height, batch_size, denoise=1.0, image=None):
        image, width, height, batch_size = _prepare_pixel_image(image, width, height, batch_size)
        if image is not None and denoise <= 0.0:
            return io.NodeOutput(image)
        latent_format = model.get_model_object("latent_format")
        if latent_format.latent_channels != 3 or latent_format.spacial_downscale_ratio != 1 or latent_format.latent_dimensions != 2:
            raise ValueError("Iris KSampler requires a pixel-space RGB model.")
        samples = _pixel_samples(image, width, height, batch_size)
        latent = {"samples": samples}
        if denoise <= 0.0:
            return io.NodeOutput(_pixel_output(samples))
        if scheduler == "iris":
            noise = comfy.sample.prepare_noise(samples, seed)
            sigmas = iris_sigmas(steps, denoise)
            sampled = {"samples": comfy.sample.sample(model, noise, steps, cfg, sampler_name, "simple", positive, negative, samples, denoise=denoise, sigmas=sigmas, callback=latent_preview.prepare_callback(model, steps), disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed)}
        else:
            sampled = nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=denoise)[0]
        return io.NodeOutput(_pixel_output(sampled["samples"]))


class IrisSamplerCustomAdvanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="IrisSamplerCustomAdvanced", display_name="Iris Sampler Custom Advanced (Pixels)", category="sampling/iris", description="Use ComfyUI custom noise, guider, sampler and sigmas to sample RGB pixels directly.", inputs=[
            io.Noise.Input("noise"), io.Guider.Input("guider"), io.Sampler.Input("sampler"), io.Sigmas.Input("sigmas"),
            io.Int.Input("width", default=1024, min=1, max=nodes.MAX_RESOLUTION), io.Int.Input("height", default=1024, min=1, max=nodes.MAX_RESOLUTION),
            io.Int.Input("batch_size", default=1, min=1, max=4096),
            io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01, optional=True, tooltip="Retains the final fraction of supplied sigma steps. Set upstream scheduler denoise to 1 to avoid trimming twice."),
            io.Image.Input("image", optional=True, tooltip="Sets dimensions and batch size; dimensions round to the nearest 16 pixels."),
        ], outputs=[io.Image.Output(display_name="output"), io.Image.Output(display_name="denoised_output")])

    @classmethod
    def execute(cls, noise, guider, sampler, sigmas, width, height, batch_size, denoise=1.0, image=None):
        image, width, height, batch_size = _prepare_pixel_image(image, width, height, batch_size)
        steps = len(sigmas) - 1
        if 0.0 < denoise < 1.0:
            steps = round(steps * denoise)
            if steps > 0:
                sigmas = sigmas[-(steps + 1):].clone()
        if denoise <= 0.0 or steps < 1:
            output = image if image is not None else _pixel_output(_pixel_samples(None, width, height, batch_size))
            return io.NodeOutput(output, output)
        latent_format = guider.model_patcher.get_model_object("latent_format")
        if latent_format.latent_channels != 3 or latent_format.spacial_downscale_ratio != 1 or latent_format.latent_dimensions != 2:
            raise ValueError("Iris Sampler Custom Advanced requires a pixel-space RGB model.")
        latent = {"samples": _pixel_samples(image, width, height, batch_size)}
        output, denoised = SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent).result
        return io.NodeOutput(_pixel_output(output["samples"]), _pixel_output(denoised["samples"]))


def _prepare_pixel_image(image, width, height, batch_size):
    if image is not None:
        batch_size, height, width = image.shape[:3]
    width, height = max(16, ((width + 8) // 16) * 16), max(16, ((height + 8) // 16) * 16)
    if image is not None and image.shape[1:3] != (height, width):
        image = comfy.utils.common_upscale(image.movedim(-1, 1), width, height, "bilinear", "disabled").movedim(1, -1)
    return image, width, height, batch_size


def _pixel_samples(image, width, height, batch_size):
    device = comfy.model_management.intermediate_device()
    return torch.zeros((batch_size, 3, height, width), device=device) if image is None else (2.0 * image[..., :3] - 1.0).movedim(-1, 1).to(device=device)


def _pixel_output(samples):
    return (samples.movedim(1, -1) + 1.0) / 2.0


def iris_sigmas(steps, denoise=1.0):
    if denoise <= 0.0:
        return torch.empty(0, dtype=torch.float32)
    total_steps = steps if denoise > 0.9999 else int(steps / denoise)
    sigma = 1.0 - torch.linspace(1.0, 0.001, total_steps + 1, dtype=torch.float64)
    sigmas = (4.0 * sigma / (1.0 + 3.0 * sigma)).flip(0).to(torch.float32)
    return sigmas if total_steps == steps else sigmas[-(steps + 1):].clone()
