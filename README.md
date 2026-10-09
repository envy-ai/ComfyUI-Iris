# ComfyUI Iris

Native text-to-image support for [Iris-3B](https://huggingface.co/speridlabs/iris-3b).
Iris predicts RGB pixels directly. Its model and text encoder use ComfyUI's
operations, attention, memory management and model patchers; no VAE or diffusers
pipeline is used. No files are downloaded during node import or execution.
The loaders select Iris's DiT and exact 12-layer, 300-position conditioning
through native ComfyUI loading and memory management, without changing core files.

Place this directory under `ComfyUI/custom_nodes`. Requires a recent ComfyUI with
native Qwen3-VL-4B support and Comfy Kitchen weighted RMSNorm/RoPE and GQA
attention. Uses only existing ComfyUI dependencies. After installation, restart
ComfyUI to register the nodes once active jobs are finished.

Model files stay in configured ComfyUI model folders:

- `diffusion_models/iris-3b/model.safetensors`: original text-to-image checkpoint.
- `text_encoders/qwen3vl_4b_bf16.safetensors`: Qwen3-VL-4B language weights.
  Compatible fine-tunes, including heretic variants, are accepted. Native
  ComfyUI FP8-scaled and INT8 ConvRot checkpoints retain their quantization.
  Supported namespaces are `model.language_model.*`, `language_model.*` and
  native `model.*`; unused vision and language output heads are ignored.
  The encoder must keep Qwen3-VL-4B's 36 layers, 2560
  hidden width and tokenizer. The original encoder is the release reference;
  changed language weights may change prompt conditioning and image quality.

Connect **Load Iris Model** to **Iris KSampler (Pixels)**. Connect **Load Iris
CLIP** to two standard **CLIP Text Encode (Prompt)** nodes for positive and
negative conditioning. Connect the sampler's IMAGE output directly to **Save
Image**. The included [API workflow](examples/text_to_image_api.json) shows these
connections and local filenames.

Optionally connect an IMAGE to **Iris KSampler (Pixels)** for image-to-image.
Its dimensions and batch size replace the width, height and batch widgets.
The sampler rounds dimensions to the nearest positive multiple of 16, rounding
halfway values up; an input image is resized with bilinear interpolation when
needed. `denoise` defaults to 1; lower values preserve more of the source using
ComfyUI's partial schedule semantics. Zero returns the image after dimension
rounding; an already aligned image stays exactly unchanged.

**Iris Sampler Custom Advanced (Pixels)** accepts standard NOISE, GUIDER,
SAMPLER and SIGMAS inputs and returns two IMAGEs: `output` and `denoised_output`,
with the same meanings as ComfyUI's Sampler Custom Advanced. It uses the same
dimension rounding, optional IMAGE and batch behavior as Iris KSampler.
Its `denoise` retains the final `round(sigma_steps * denoise)` steps of the
supplied schedule. Set the upstream scheduler's denoise to 1 when using this
input to avoid trimming twice. Full denoise leaves supplied sigmas unchanged;
zero denoise or an empty sampling schedule returns the prepared image on both
outputs without generating noise. No VAE is needed.

Defaults follow the release: 1024×1024, 100 steps, CFG 3, `dpmpp_2m`. The `iris`
scheduler uses upstream's shifted rectified-flow grid (shift 4, unshifted starting
sigma 0.999, final sigma 0). Other ComfyUI samplers and schedulers are available;
results can differ from upstream's solver. Seeds follow ComfyUI's noise generator.
The sampler supports other RGB pixel-space MODELs when their latent format has
three channels and no spatial downsampling; use their appropriate scheduler.

Standard **KSampler** also works with the Iris MODEL and
**EmptyChromaRadianceLatentImage**. Its LATENT contains RGB in the model's
approximately −1…1 domain; convert to IMAGE with `(pixels + 1) / 2`, preserving
NHWC ordering. The dedicated sampler performs this conversion and returns IMAGE.
The Iris MODEL accepts dimensions that are not divisible by 16, pads internally
and crops to the supplied latent size. Encoding keeps the release's 300 positions, chat suffix, attention mask,
and 12 raw Qwen hidden layers. The text adapter runs once per conditioning.

This package implements text-to-image. The separate `depth/model.safetensors`
and `upscaler/model.safetensors` checkpoints require their one-step downstream
recipes and are not supported by these nodes.

The model port is derived from Speridlabs source revision
`a8d15239dea469aba042cfa56ca3bb4e450d5ebc`. Original Apache-2.0 LICENSE and NOTICE
are included. ComfyUI and the Qwen weights keep their respective licenses.

Source compilation passed. Before the custom advanced node was added, the
focused CPU suite passed (26 tests). Tests in `tests/`
cover tiny-model layout and upstream parity, loader boundaries, compatible and
quantized CLIP loading, tokenizer behavior and the IMAGE sampler contract.
Real FP16 and corrected mixed INT8 Iris checkpoints both generated finite 64×64
images with the INT8 heretic encoder on RTX 4090: one Euler step, CFG 1.5, native
Comfy Kitchen attention (`--use-ck-attention`). Encoder states and masks matched
the required layouts. These checks do not establish image quality at release settings.
Real checkpoint image-to-image verification is pending while training is active.
Custom advanced sampler tests are authored and pending the same training/queue
condition. The new node loads on the next ComfyUI restart.

On the tested local PyTorch/cuDNN build, PyTorch attention fails in the masked
text adapter with `No valid execution plans built`. Use Comfy Kitchen attention
in that environment; this package keeps ComfyUI's selected backend.

## Local precision conversions

The model loader accepts the original checkpoint, FP16 storage, and ComfyUI's
native mixed INT8 ConvRot format. To make either smaller checkpoint on CPU from
the original file:

```bash
conda run -n comfyui python custom_nodes/ComfyUI-Iris/scripts/quantize.py \
  /d/comfy_models/diffusion_models/iris-3b/model.safetensors \
  /d/comfy_models/diffusion_models/iris-3b/Iris-3B_fp16.safetensors --format fp16
conda run -n comfyui python custom_nodes/ComfyUI-Iris/scripts/quantize.py \
  /d/comfy_models/diffusion_models/iris-3b/model.safetensors \
  /d/comfy_models/diffusion_models/iris-3b/Iris-3B_int8_convrot.safetensors --format int8-convrot
```

The converter uses CPU only, defaults to two threads, streams one source tensor
at a time, preserves source metadata, and rejects non-finite source/conversion
values. It refuses existing output files and publishes completed artifacts from
`.partial` files. An adjacent JSON records sizes, tensor counts and policy.

The mixed checkpoint stores 128 trunk attention Q/K/V/output projections as INT8
with per-output-channel FP32 scales. ConvRot uses
256-column groups, matching the native fused activation-rotation path. Content
gate matrices, all MLPs, the text adapter, modulation, timestep paths,
norms, input/output layers and the complete pixel head stay FP16. The w2 width
is 6826, which cannot use these rotation groups. The w1/w3 output width is also
6826, incompatible with the native GPU INT8 GEMM's multiple-of-four requirement.
This is a conservative initial precision policy; image quality and GPU kernel
performance have not been evaluated.
