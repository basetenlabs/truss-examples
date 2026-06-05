# Qwen3-VL-32B Instruct — Throughput Template

Qwen3-VL-32B-Instruct (NVFP4) is a multimodal vision-language model served with vLLM on a single RTX PRO 6000 (Blackwell). It exposes an OpenAI-compatible chat completions endpoint with image input support.

## Requirements

- `truss >= 0.10.5`
- Baseten account with RTX PRO 6000 GPU access
- `hf_access_token` secret set in your Baseten account

## Deployment

```sh
git clone https://github.com/basetenlabs/model-registry.git
cd model-registry/llm/qwen3-vl-32b/throughput
truss push --trusted --publish
```