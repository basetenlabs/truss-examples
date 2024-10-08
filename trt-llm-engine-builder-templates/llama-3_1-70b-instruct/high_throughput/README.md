# Llama 3.1 70B Instruct using TensorRT-LLM with High Throughput

This directory is [Truss](https://truss.baseten.co/) template for deploying model Llama 3.1 70B Instruct using our TensorRT-LLM (TRT-LLM) [engine builder](https://docs.baseten.co/performance/engine-builder-overview). This configuration is optimized for high-throughput scenarios.

## Use case

This deployment is tailored for applications that require processing large volumes of data with moderate to low latency, such as:
* Content moderation for social media platforms
* Bulk article summarization
* Large-scale Retrieval-Augmented Generation (RAG) systems

## Configuration

The template uses the following key configuration parameters:

| Property            | Value    | Description                                                                    |
| ------------------- | -------- | ------------------------------------------------------------------------------ |
| GPU                 | 2xH100   | Two NVIDIA H100 GPUs                                                           |
| `max_batch_size`    | 16       | Allows processing up to 16 requests simultaneously                             |
| `quantization_type` | `fp8_kv` | FP8 quantization, balancing performance and accuracy. See this [blog](https://www.baseten.co/blog/33-faster-llm-inference-with-fp8-quantization/). |
| `max_input_len`     | 4096     | Maximum number of input tokens that the model will accept                      |
| `max_output_len`    | 1024     | Maximum number of output tokens the model can generate                         |

## Performance Metrics

A preliminary benchmark was conducted with the following parameters:
- 150 total requests
- 16 concurrent requests
- ~4000 input tokens per request

Results:

| Metric                              | Value              |
| ----------------------------------- | ------------------ |
| Average Latency                     | 2.6890 s           |
| Average Time to First Token (TTFT)  | 2.6097 s           |
| Average Perceived Tokens per Second | 281.4879           |
| Average Overall Throughput          | 4503.8064 tokens/s |

## Deployment

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd trt-llm-engine-builder-templates/llama-3_1-70b-instruct/high_throughput
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `trt-llm-engine-builder-templates/llama-3_1-70b-instruct/high_throughput` as your working directory, you can deploy the model with:

```sh
truss push --trusted --publish
```

Paste your Baseten API key if prompted. Also ensure the `hf_access_token` secret is properly setup in your Baseten Account to access this model.

**Note**: TensorRT-LLM with engine builder will only work under a Baseten production deployment

For more information, refer to the [Truss documentation](https://docs.baseten.co/performance/engine-builder-overview).
