# GLM-5.3-Flash onboarding notes

- Serve the native FP8 checkpoint with the dedicated, digest-pinned vLLM
  `glm53-flash` image. Stable public vLLM images do not yet contain this model's
  integration.
- Pin this preset to one `H100:8` instance with tensor parallel size 8. The
  checkpoint is about 306 GiB before runtime and KV-cache overhead.
- Do not enable FP8 KV cache on H100. The current GLM-5.3-Flash implementation
  supports FP8 KV cache on Blackwell, while Hopper must use BF16 KV cache.
- Keep `--no-enable-flashinfer-autotune` on Hopper, following the official vLLM
  hardware override.
- BIS-LLM is not selected for this onboarding. Although the model is a large MoE,
  no Baseten-provided BIS version or GPU image is established for the new
  `glm5_next` architecture, and the official deployment recipe is the dedicated
  vLLM image. Reconsider BIS only when an explicitly compatible Baseten image and
  version are available.
