Notes for FDE:

## Model summary

* Krea 2 Turbo is Krea's 12B flow-matching image generation model: a
  single-stream MMDiT backbone paired with a Qwen3-VL text encoder and the
  Qwen-Image VAE.
* The Turbo variant is TDM-distilled to 8 denoising steps with CFG disabled.
  Do not pass `num_inference_steps` or guidance parameters in requests; the
  Turbo defaults (8 steps, CFG off) are baked into the SGLang model registry.
* Native resolution range is 1024 to 2048. Expect roughly 1.5 to 2 seconds
  per 1024px image on H100-class hardware.

## Serving stack

* Served by the Baseten SGLang fork (basetenlabs/sglang, branch `b10-main`,
  commit aff5ff91 or later) via the OpenAI-compatible images API
  (`/v1/images/generations`).
* Base image `baseten/sglang-diffusion-h100:v1.3` is the first tag with
  Krea-2 support.
* Weights are ~34GB in bf16 and fit resident on a single H100 80GB, so all
  CPU offload flags and CPU memory saving are disabled.
* The HuggingFace repo `krea/Krea-2-Turbo` is not gated; the
  `hf_access_token` secret is kept for consistency with sibling entries.

## LICENSE

* Krea 2 is released under the Krea 2 Community License. Commercial use by
  companies with more than $1M in company-wide annual revenue requires an
  Enterprise License from Krea (opensource@krea.ai).