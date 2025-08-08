# GPT OSS 120B with TensorRT-LLM (TorchFlow) — High-Throughput Template

GPT OSS 120B is OpenAI's open source model designed for powerful reasoning, agentic tasks and other developer use cases.

This directory contains a **[Truss](https://truss.baseten.co/)** template for deploying **GPT OSS 120B** with Baseten’s **TensorRT-LLM (TRT-LLM) + PyTorch backend** stack on 2 H100 GPUs. This inference stack maximizes both inference and throughput.

---


## Core TRT-LLM `runtime` parameters

| Property (YAML path)  | Value                | Why it matters |
| --------------------- | -------------------- | -------------- |
| `tensor_parallel_size`| **2** | Shards every weight matrix across the 2 H100s |
| `moe_expert_parallel_size` | **2** | Shards each expert across 2 H100s |
| `max_batch_size`      | **64** | Up to 64 concurrent requests per forward pass |
| `max_seq_len`         | **98304** | 96 context length |
| `enable_chunked_prefill` | `true` | Streams very long prompts without bursting VRAM |
| `max_num_tokens`      | **8192** | Upper limit on total tokens per chunk |
| `served_model_name`   | `openai/gpt-oss-120b` | `model: openai/gpt-oss-120b` to call this model in OpenAI Compatible server |

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

| Metric                              | Value              |
| ----------------------------------- | ------------------ |
| Average Latency                     | NA           |
| Average Time to First Token (TTFT)  | NA           |
| Average Perceived Tokens per Second | NA           |
| Average Overall Throughput          | NA           |

---

## Deployment

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd torchflow-templates/gpt-oss-120b
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `torchflow-templates/gpt-oss-120b` as your working directory, you can deploy the model with:

```sh
truss push --trusted --publish
```

Paste your Baseten API key if prompted. Also ensure the `hf_access_token` secret is properly setup in your Baseten Account to access this model.

**Note**: TensorRT-LLM with PyTorch Backend will only work under a Baseten production deployment

For more information, refer to the [Truss documentation](https://docs.baseten.co/performance/engine-builder-overview).
