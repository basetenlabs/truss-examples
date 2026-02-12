# Llama 3.1 8B Instruct using TensorRT-LLM with Low TTFT

This directory is a [Truss](https://truss.baseten.co/) template for deploying model Llama 3.1 8B Instruct using our TensorRT-LLM (TRT-LLM) [engine builder](https://docs.baseten.co/performance/engine-builder-overview). This configuration is optimized for low Time to First Token (TTFT) scenarios.

## Use case

This deployment is tailored for applications that require rapid response times, such as:
* Real-time chat assistants
* Voice assistance
* Code editor auto-complete
* Translation services

## Configuration

The template uses the following key configuration parameters:

| Property         | Value  | Description                                                                                        |
| ---------------- | ------ | -------------------------------------------------------------------------------------------------- |
| GPU              | 1xH100 | Single NVIDIA H100 GPU                                                                             |
| `max_batch_size` | 8      | Allows processing up to 8 requests simultaneously                                                  |
| `quantization_type` | `fp8_kv` | FP8 quantization, balancing performance and accuracy. See this [blog](https://www.baseten.co/blog/33-faster-llm-inference-with-fp8-quantization/). |
| `max_input_len`  | 4096   | Maximum number of input tokens that the model will accept                                          |
| `max_output_len` | 1024   | Maximum number of output tokens the model can generate                                             |
| `prefix_caching` | `true` | Reuse KV Cache across requests, improving performance when requests share the same prompt prefixes |

## Performance Metrics

A preliminary benchmark was conducted with the following parameters:
- 150 total requests
- 8 concurrent requests
- ~4000 input tokens per request

Results:

| Metric                              | Value              |
| ----------------------------------- | ------------------ |
| Average Latency                     | 1.2478 s           |
| Average Time to First Token (TTFT)  | 0.7818 s           |
| Average Perceived Tokens per Second | 1177.5856          |
| Average Overall Throughput          | 9420.6847 tokens/s |

## Deployment

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd trt-llm-engine-builder-templates/llama-3_1-8b-instruct/low_ttft
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `trt-llm-engine-builder-templates/llama-3_1-8b-instruct/low_ttft` as your working directory, you can deploy the model with:

```sh
truss push --trusted --publish
```

Paste your Baseten API key if prompted. Also ensure the `hf_access_token` secret is properly setup in your Baseten Account to access this model.

**Note**: TensorRT-LLM with engine builder will only work under a Baseten production deployment

For more information, refer to the [Truss documentation](https://docs.baseten.co/performance/engine-builder-overview).
