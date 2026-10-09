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

Implementation is complete; runtime verification is pending the queue/training
condition. The package exposes three nodes (MODEL loader, native CLIP loader,
direct IMAGE KSampler), an exact upstream sigma grid, and an API example.
Static source compilation passed in the comfyui environment. Static review and
checkpoint header inspection found the released shape/key contract aligned.

Optimized math decisions: paired weighted RMSNorm+RoPE uses Comfy Kitchen's
`rms_rope` with native parameter casting and cleanup. Native attention owns GQA
backend dispatch. The existing HiDream SwiGLU structure is reused with its
exact width rule and ComfyUI's fused linear-input activation helper. Iris uses
exact GELU; the fused linear helper supports GELU-tanh only, so its pixel MLP
keeps native Linear + exact F.gelu. Fused `rms_adaln` lacks Iris's learned
RMSNorm affine weight, so native RMSNorm plus compound addcmul keeps that math.
