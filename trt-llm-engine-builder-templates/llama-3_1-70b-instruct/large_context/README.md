# Llama 3.1 70B Instruct using TensorRT-LLM for Large Context

This directory is [Truss](https://truss.baseten.co/) template for deploying model Llama 3.1 70b Instruct using our TensorRT-LLM (TRTLLM) [engine builder](https://docs.baseten.co/performance/engine-builder-overview). This configuration is optimized for large context scenarios.

## Use case

This deployment is tailored for applications that require processing extensive amounts of data with high accuracy, such as:
* Textbook summarization
* Codebase analysis
* Multi-turn conversation

## Configuration

The template uses the following key configuration parameters:

| Property            | Value  | Description                                       |
| ------------------- | ------ | ------------------------------------------------- |
| GPU                 | 2xH100 | Two NVIDIA H100 GPUs                              |
| `max_batch_size`    | 8      | Allows processing up to 8 requests simultaneously |
| `quantization_type` | `fp16` | FP16 quantization to retain model accuracy        |
| `max_input_len`     | 8192   | Maximum number of input tokens                    |
| `max_output_len`    | 4096   | Maximum number of output tokens                   |

## Performance Metrics

A preliminary benchmark was conducted with the following parameters:
- 150 total requests
- 8 concurrent requests
- ~8000 input tokens per request

Results:

| Metric                              | Value              |
| ----------------------------------- | ------------------ |
| Average Latency                     | 3.6395 s           |
| Average Time to First Token (TTFT)  | 2.6284 s           |
| Average Perceived Tokens per Second | 404.0099           |
| Average Overall Throughput          | 3232.0791 tokens/s |

## Deployment

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd trt-llm-engine-builder-templates/llama-3_1-70b-instruct/large_context
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `trt-llm-engine-builder-templates/llama-3_1-70b-instruct/large_context` as your working directory, you can deploy the model with:

```sh
truss push --trusted --publish
```

Paste your Baseten API key if prompted. Also ensure the `hf_access_token` secret is properly setup in your Baseten Account to access this model.

**Note**: TensorRT-LLM with engine builder will only work under a Baseten production deployment

For more information, refer to the [Truss documentation](https://docs.baseten.co/performance/engine-builder-overview).