# GPT OSS 120B with vLLM — Throughput Template

GPT OSS 120B is OpenAI's largest open-source model, a 120-billion-parameter Mixture-of-Experts (MoE) architecture designed for powerful reasoning, agentic tasks, and developer use cases. It uses OpenAI's Harmony response format and tokenizer.

This directory contains a **[Truss](https://truss.baseten.co/)** template for deploying **GPT OSS 120B** with **vLLM** on **4x B200 GPUs**. Unlike TRT-LLM-based templates, this uses a Docker Server approach with the official `vllm/vllm-openai` container image, configured for maximum throughput on Blackwell hardware.

---

## Requirements

- `truss >= 0.10.5`
- Baseten account with B200 GPU access
- `hf_access_token` secret set in your Baseten account (for gated model access)

---

## Key Configuration

| Parameter | Value | Why it matters |
| --- | --- | --- |
| `base_image` | `vllm/vllm-openai:v0.12.0` | Pinned vLLM image for reproducibility |
| `accelerator` | `B200:4` | 4x Blackwell B200 GPUs |
| `tensor-parallel-size` | `4` | Shards model across all 4 GPUs |
| `gpu-memory-utilization` | `0.95` | Maximizes available VRAM for KV cache |
| `max-model-len` | `8192` | Context window (increase for longer sequences at the cost of throughput) |
| `max-num-seqs` | `256` | Up to 256 concurrent sequences per forward pass |
| `kv-cache-dtype` | `fp8` | FP8 KV cache for higher throughput on Blackwell |
| `predict_concurrency` | `256` | Allows 256 concurrent requests at the Truss level |
| `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8` | `1` | Enables FlashInfer MXFP4+MXFP8 MoE kernels for Blackwell |

### Build-time Optimizations

The `build_commands` pre-download Harmony/tiktoken vocabulary files so the runtime environment does not need network access for tokenizer initialization. The environment variables `TIKTOKEN_ENCODINGS_BASE` and `TIKTOKEN_RS_CACHE_DIR` point vLLM to these cached files.

### vLLM Compilation Config

The `--compilation-config` flag enables two optimization passes:
- **`fuse_allreduce_rms`**: Fuses AllReduce and RMSNorm operations to reduce kernel launch overhead in tensor-parallel setups.
- **`eliminate_noops`**: Removes no-op operations from the computation graph.

---

## Deployment

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/model-registry.git
cd model-registry/llm/gpt-oss-120b/throughput
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`
3. Set up your `hf_access_token` secret in [Baseten settings](https://app.baseten.co/settings/secrets).

Deploy with:

```sh
truss push --trusted --publish
```

Paste your Baseten API key if prompted.

---

## Call your model

### OpenAI-compatible inference

This deployment exposes an OpenAI-compatible `/v1/chat/completions` endpoint.

```python
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.environ["BASETEN_API_KEY"],
    base_url="https://model-xxxxxx.api.baseten.co/environments/production/sync/v1",
)

response = client.chat.completions.create(
    model="gpt-oss-120b",
    messages=[
        {"role": "user", "content": "Explain the difference between MoE and dense transformers."}
    ],
    stream=True,
    max_tokens=2048,
    temperature=0.5,
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

---

## Support

If you have any questions or need assistance, please open an issue in this repository or contact our support team.