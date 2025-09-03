# DeepSeek-V3 0324 with BISv2 — High-Throughput Template

DeepSeek V3 is a powerful model that excels in coding, mathematical reasoning and ideal for building agents. Especially in the world of building agents, latencies matter.

This directory contains a **[Truss](https://truss.baseten.co/)** template for deploying **DeepSeek-V3 0324 (FP4)** with **Baseten Inference Stack v2 (TensorRT-LLM + PyTorch backend)** on 8 B200 GPUs. This inference stack maximizes both inference and throughput.

---


## Core TRT-LLM `runtime` parameters

| Property (YAML path)  | Value                | Why it matters |
| --------------------- | -------------------- | -------------- |
| `tensor_parallel_size`| **8** | Shards every weight matrix across the 8 B200s |
| `moe_expert_parallel_size` | **4** | Shards each expert across 4 B200s |
| `max_batch_size`      | **64** | Up to 64 concurrent requests per forward pass |
| `max_seq_len`         | **98304** | 96 k-token context length supported by DeepSeek |
| `enable_chunked_prefill` | `true` | Streams very long prompts without bursting VRAM |
| `max_num_tokens`      | **8192** | Upper limit on total tokens per chunk |
| `served_model_name`   | `deepseek-ai/DeepSeek-V3-0324` | `model: deepseek-ai/Deepseek-V3-0324` to call this model in OpenAI Compatible server |

---

## Important Advanced **`runtime.patch_kwargs`** parameters

These map 1-to-1 to TensorRT-LLM flags for extra performance tuning.

| Property (YAML path)                    | Value / Setting | Effect |
| --------------------------------------- | --------------- | ------ |
| `speculative_config.decoding_type`      | `MTP`           | Enables Multi-Token Prediction speculative decoding |
| `speculative_config.num_nextn_predict_layers` | **3** | Draft uses first L-3 layers → cheaper guesses |
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
| Average Latency                     | 3.4553s           |
| Average Time to First Token (TTFT)  | 0.4397s           |
| Average Perceived Tokens per Second | 1304.2998           |
| Average Overall Throughput          | 20868.7972 tokens/s |

---

## Deployment

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd trt-llm-torch-templates/deepseek-v3-0324
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `trt-llm-torch-templates/deepseek-v3-0324` as your working directory, you can deploy the model with:

```sh
truss push --trusted --publish
```

Paste your Baseten API key if prompted. Also ensure the `hf_access_token` secret is properly setup in your Baseten Account to access this model.

**Note**: TensorRT-LLM with PyTorch Backend will only work under a Baseten production deployment

For more information, refer to the [Truss documentation](https://docs.baseten.co/performance/engine-builder-overview).
