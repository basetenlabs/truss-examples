# Qwen3 Coder 30B Instruct with BISv2 — High-Throughput Template

Qwen3 Coder 30B is a MoE model that is an expert in reasoning, instruction-following, human preference alignment, and agent capabilities.

This directory contains a **[Truss](https://truss.baseten.co/)** template for deploying **Qwen3 Coder 30B Instruct** with **Baseten Inference Stack v2 (TensorRT-LLM + PyTorch backend)** on 8 B200 GPUs. This inference stack maximizes both inference and throughput.

---


## Core TRT-LLM `runtime` parameters

| Property (YAML path)  | Value                | Why it matters |
| --------------------- | -------------------- | -------------- |
| `tensor_parallel_size`| **4** | Shards every weight matrix across the 4 B200s |
| `max_batch_size`      | **16** | Up to 16 concurrent requests per forward pass |
| `max_seq_len`         | **98304** | Max context length |
| `served_model_name`   | `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` | `model: Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` to call this model in OpenAI Compatible server |

---

## Important Advanced **`runtime.patch_kwargs`** parameters

These map 1-to-1 to TensorRT-LLM flags for extra performance tuning.

| Property (YAML path)                    | Value / Setting | Effect |
| --------------------------------------- | --------------- | ------ |
| `cuda_graph_config.enable_padding`      | `true`          | Pad to fixed shape so one CUDA Graph is reused every step |
| `kv_cache_config.free_gpu_memory_fraction` | **0.8** | 80 % of post-load VRAM reserved for paged KV-cache |
| `kv_cache_config.enable_block_reuse`    | `true`          | Identical prefixes share cache blocks → faster TTFT |

---

## Performance Metrics

A preliminary benchmark was conducted with the following parameters:

- 150 total requests
- 16 concurrent requests
- ~4000 input tokens per request

Results:

| Metric                              |Torchflow              |vLLM               |
| ----------------------------------- | ------------------ | ---------------- |
| Average Latency                     | 7.3772 s           | 8.2202s           |
| Average Time to First Token (TTFT)  | 0.2066 s           | 0.1888s           |
| Average Perceived Tokens per Second | 974.0529           | 908.9224            |

---

## Deployment

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd baseten-inference-stack-v2-templates/qwen3-Coder-30b-instruct
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `baseten-inference-stack-v2-templates/qwen3-Coder-30b-instruct` as your working directory, you can deploy the model with:

```sh
truss push --trusted --publish
```

Paste your Baseten API key if prompted. Also ensure the `hf_access_token` secret is properly setup in your Baseten Account to access this model.

**Note**: TensorRT-LLM with PyTorch Backend will only work under a Baseten production deployment

For more information, refer to the [Truss documentation](https://docs.baseten.co/performance/engine-builder-overview).
