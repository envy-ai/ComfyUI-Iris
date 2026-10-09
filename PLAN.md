# Native Iris support

Scope: standalone ComfyUI custom nodes for the released Iris-3B text-to-image
checkpoint. No changes to core model detection or unrelated custom nodes.
The depth and restoration checkpoints share the architecture but their one-step
inference flows are outside this initial text-to-image integration.

Acceptance: load local safetensors through ComfyUI memory/offload policy; encode
Qwen3-VL-4B's 12 selected hidden layers with the exact Iris chat template and
padding; use native MODEL and CONDITIONING with ComfyUI samplers; expose a
KSampler that returns RGB IMAGE without a VAE; preserve supplied dimensions by
padding model inputs to patch boundaries and cropping the output.

Implementation order:
1. Port only released inference architecture, keeping checkpoint keys. Use
   ComfyUI operations, selected optimized attention (including GQA), and shared
   Comfy Kitchen RoPE; scope position tables to each forward.
2. Integrate BaseModel, rectified-flow sampling (shift 4), model patcher and
   local file resolvers. Preprocess the timestep-independent text adapter once
   per conditioning through BaseModel.extra_conds.
3. Extend the native Qwen3-VL text stack for selected raw hidden layers; return
   CLIP usable by CLIPTextEncode. Reuse existing pixel LATENT where applicable.
4. Document local files and provide a minimal workflow. Verify small-model
   upstream parity, dtype/offload and odd-size handling, then real checkpoint
   load/generation after the coordinator confirms training and queues empty.

Dependencies: existing ComfyUI and its installed torch, transformers, PyYAML,
safetensors. No diffusers pipeline, upstream runtime package, or downloads at
node import/execution. Upstream source pinned to
`a8d15239dea469aba042cfa56ca3bb4e450d5ebc` (Apache-2.0); retain LICENSE/NOTICE.

Risks: selected Qwen hidden-layer numbering must match pre-block native taps
and post-block Hugging Face taps; padded text must remain in the trunk; Iris
velocity uses decreasing noise time and ComfyUI CONST conventions. Pixel-stage
activations are large at 1024 px, so use native offloading and conservative
memory estimates. Required tests must not run while user's training/queues run.

Implementation is complete. After training and queues emptied, the focused CPU
suite passed (26 tests). FP16 and corrected mixed INT8 checkpoints both passed
64×64, one-step Euler generation with the INT8 heretic encoder and native Comfy
Kitchen attention on RTX 4090. The package
exposes four nodes (MODEL loader, native CLIP loader, direct IMAGE KSampler and
custom advanced IMAGE sampler), an exact upstream sigma grid, and an API example.
Static source compilation passed in the comfyui environment. Static review and
checkpoint header inspection found the released shape/key contract aligned.

Optimized math decisions: paired weighted RMSNorm+RoPE uses Comfy Kitchen's
`rms_rope` with native parameter casting and cleanup. Native attention owns GQA
backend dispatch. The existing HiDream SwiGLU structure is reused with its
exact width rule and ComfyUI's fused linear-input activation helper. Iris uses
exact GELU; the fused linear helper supports GELU-tanh only, so its pixel MLP
keeps native Linear + exact F.gelu. Fused `rms_adaln` lacks Iris's learned
RMSNorm affine weight, so native RMSNorm plus compound addcmul keeps that math.

## CPU precision conversions

Add a one-tensor-at-a-time converter for the main original checkpoint. FP16 casts
all floating checkpoint tensors, rejecting values that become non-finite. The
mixed INT8 ConvRot policy quantizes 128 trunk Q/K/V and attention output projections
with per-output-channel FP32 scales and group size 256. Keep content gate matrices,
all MLPs, the text adapter, all modulation/time paths,
input/output layers, norms and the pixel head in FP16. The w2 input width is 6826,
which cannot use the 256-column rotation grouping without altering the model.
The w1/w3 output width is also 6826, outside the native GPU INT8 GEMM's
multiple-of-four requirement. Eligibility checks input width modulo 256 and
output width modulo four; model/checkpoint shapes remain unchanged.
This policy is an informed initial choice, without image-quality calibration.

Store native `.comfy_quant` metadata and select ComfyUI's mixed-precision ops at
load. Conversion runs only on CPU, with bounded threads, and does not instantiate
or run a model. Write `.partial` files before committing completed artifacts;
leave the original checkpoint untouched. Preserve upstream file metadata and
record conversion parameters/tensor counts in the file and an adjacent JSON.
CPU artifact conversion was separately authorized during training.

Compatible Qwen3-VL-4B fine-tunes and native quantized language weights use the
same encoder path. Normalize native quant metadata and supported language
namespaces, validate architecture and required weights, then load once through
ComfyUI's CLIP constructor. Keep the exact Iris conditioning recipe.

Add optional IMAGE and denoise inputs at the sampler boundary, preserving old
positional calls and default text-to-image behavior. IMAGE owns dimensions and
batch size. At the user's request, round sampler dimensions to the nearest
positive multiple of 16 and resize supplied IMAGEs with bilinear interpolation
when needed. Normalize RGB to the existing pixel latent range.
Use native denoise forwarding and slice the Iris sigma grid like KSampler;
zero denoise returns the rounded IMAGE (aligned inputs stay exactly unchanged).
Focused sampler, rounding/resizing and model odd-size padding/cropping tests
passed after fresh queue/training clearance. Real checkpoint image-to-image
verification is pending because training restarted; no further tests run while
training or queues are active.
Model math and its arbitrary-size padding/cropping remain unchanged.

The custom advanced sampler delegates execution to ComfyUI's existing
SamplerCustomAdvanced, preserving noise, guider, callbacks and denoised output.
Share pixel input preparation with IrisKSampler. Accept external SIGMAS unchanged
at full denoise; partial denoise retains a cloned tail using SplitSigmasDenoise's
rounded step count. Zero effective steps return the prepared image without
sampling. New tests remain unexecuted while training is active; no restart is
performed because the user plans a clean terminal restart.
