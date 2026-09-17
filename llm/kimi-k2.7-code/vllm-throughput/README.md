# Kimi K2.7 Code

This customer-deployable preset serves Moonshot AI's
[Kimi K2.7 Code](https://huggingface.co/moonshotai/Kimi-K2.7-Code) from
[NVIDIA's public NVFP4 checkpoint](https://huggingface.co/nvidia/Kimi-K2.7-Code-NVFP4)
using vLLM on eight B200 GPUs.

| Property | Value |
| --- | --- |
| Serving stack | vLLM 0.28.0 |
| Hardware | 8x B200 |
| Checkpoint | `nvidia/Kimi-K2.7-Code-NVFP4@9f28d60908c98034bb288e755566764eac77cc0d` |
| Precision | NVFP4 weights and FP8 KV cache |
| Context window | 262,144 tokens |
| Served model name | `moonshotai/Kimi-K2.7-Code` |
| Endpoint | `/v1/chat/completions` |
| Inputs | Text, image, and video |

The checkpoint is public and ungated, so customers do not need to configure a
Hugging Face access token. Vision is handled locally by the checkpoint's MoonViT
encoder; no external encoder service or API credential is required.

The NVIDIA checkpoint is governed by the NVIDIA Open Model License, and the
underlying Moonshot AI model uses the Modified MIT License.