# GPT OSS 120B with vLLM — High-Performance Template

GPT OSS 120B is OpenAI's open source model designed for powerful reasoning, agentic tasks and other developer use cases. It uses their open source response format, Harmony.

This directory contains a **[Truss](https://truss.baseten.co/)** template for deploying **GPT OSS 120B** with **vLLM** on 4 B200 GPUs. This truss fully abstracts OpenAI's harmony response format, so everything works out of the box. You can simply use it like a regular OpenAI compatible server.

---

# Requirements

`truss==0.10.5`

You also need the files downloaded during build, which include GPT's harmony encoding ahead of time, because once deployed, the deployment will be unable to download from internet.

The environment variables in `config.yaml` point `openai_harmony` to the local encoding files.

---

## Core vLLM Configuration

| Property | Value | Description |
|----------|-------|-------------|
| `accelerator` | **B200:4** | 4 B200 GPUs for optimal performance |
| `tensor_parallel_size` | **4** | Distributes model across 4 GPUs |
| `max_model_len` | **8192** | Maximum sequence length |
| `max_num_seqs` | **256** | Maximum concurrent sequences |
| `gpu_memory_utilization` | **0.95** | GPU memory utilization |
| `predict_concurrency` | **256** | Runtime concurrency setting |

---

## Important Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8` | **"1"** | Enables Blackwell optimizations |
| `TIKTOKEN_ENCODINGS_BASE` | **"/opt/tiktoken"** | Local Harmony vocab location |
| `TIKTOKEN_RS_CACHE_DIR` | **"/opt/tiktoken"** | Tiktoken cache directory |

---

## Deployment

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd openai/gpt-oss-120b-vllm
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `gpt-oss-120b-vllm` as your working directory, you can deploy the model with:

```sh
truss push --trusted --publish
```

Paste your Baseten API key if prompted. Also ensure the `hf_access_token` secret is properly setup in your Baseten Account to access this model.

For more information, refer to the [Truss documentation](https://docs.baseten.co/performance/engine-builder-overview).