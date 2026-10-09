# ComfyUI Iris

Native text-to-image support for [Iris-3B](https://huggingface.co/speridlabs/iris-3b).
Iris predicts RGB pixels directly. Its model and text encoder use ComfyUI's
operations, attention, memory management and model patchers; no VAE or diffusers
pipeline is used. No files are downloaded during node import or execution.

Place this directory under `ComfyUI/custom_nodes`. Requires a recent ComfyUI with
native Qwen3-VL-4B support and Comfy Kitchen weighted RMSNorm/RoPE and GQA
attention. Uses only existing ComfyUI dependencies. After installation, restart
ComfyUI to register the nodes once active jobs are finished. The live instance
has not been restarted during training.

Model files stay in configured ComfyUI model folders:

- `diffusion_models/iris-3b/model.safetensors`: original text-to-image checkpoint.
- `text_encoders/qwen3vl_4b_bf16.safetensors`: original Qwen3-VL-4B-Instruct weights
  with the `model.language_model.*` namespace, as used by ComfyUI's native Qwen
  encoder. Other 4B Qwen models and altered instruction checkpoints are not the
  Iris text encoder.

Connect **Load Iris Model** to **Iris KSampler (Pixels)**. Connect **Load Iris
CLIP** to two standard **CLIP Text Encode (Prompt)** nodes for positive and
negative conditioning. Connect the sampler's IMAGE output directly to **Save
Image**. The included [API workflow](examples/text_to_image_api.json) shows these
connections and local filenames.

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
Sizes need not be divisible by 16: Iris pads internally and crops to the supplied
size. Encoding keeps the release's 300 positions, chat suffix, attention mask,
and 12 raw Qwen hidden layers. The text adapter runs once per conditioning.

This package implements text-to-image. The separate `depth/model.safetensors`
and `upscaler/model.safetensors` checkpoints require their one-step downstream
recipes and are not supported by these nodes.

The model port is derived from Speridlabs source revision
`a8d15239dea469aba042cfa56ca3bb4e450d5ebc`. Original Apache-2.0 LICENSE and NOTICE
are included. ComfyUI and the Qwen weights keep their respective licenses.

Source compilation passed. Runtime tests and real checkpoint generation are
explicitly deferred at the user's request until training and queues are empty. Tests in `tests/` include tiny-model layout and upstream parity
checks, loader boundaries, tokenizer behavior and the IMAGE sampler contract.
