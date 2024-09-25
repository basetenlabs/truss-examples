# Llama 3.1 70B Instruct using TensorRT-LLM with Low TTFT

This directory is [Truss](https://truss.baseten.co/) template for deploying model Llama 3.1 70b Instruct using our TensorRT-LLM (TRTLLM) [engine builder](https://docs.baseten.co/performance/engine-builder-overview). This configuration is optimized for low Time to First Token (TTFT) scenarios.

## Use case

This deployment is tailored for applications that require rapid response times, such as:
* Real-time chat assistants
* Voice assistance
* Code editor auto-complete
* Translation services

## Configuration

The template uses the following key configuration parameters:

| Property            | Value    | Description                                                                    |
| ------------------- | -------- | ------------------------------------------------------------------------------ |
| GPU                 | 2xH100   | Two NVIDIA H100 GPUs                                                           |
| `max_batch_size`    | 4        | Allows processing up to 4 requests simultaneously                              |
| `quantization_type` | `fp8_kv` | FP8 quantization for key and value tensors, balancing performance and accuracy |
| `max_input_len`     | 4096     | Maximum number of input tokens                                                 |
| `max_output_len`    | 1024     | Maximum number of output tokens                                                |
| `prefix_caching`    | `true`   | Enables caching of prefix computations for faster responses                    |

## Performance Metrics

A preliminary benchmark was conducted with the following parameters:
- 150 total requests
- 8 concurrent requests
- ~4000 input tokens per request

Results:

| Metric                              | Value              |
| ----------------------------------- | ------------------ |
| Average Latency                     | 1.3686 s           |
| Average Time to First Token (TTFT)  | 0.7580 s           |
| Average Perceived Tokens per Second | 530.0073           |
| Average Overall Throughput          | 2120.0290 tokens/s |

## Deployment

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd trt-llm-engine-builder-templates/llama-3_1-70b-instruct/low_ttft
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `trt-llm-engine-builder-templates/llama-3_1-70b-instruct/low_ttft` as your working directory, you can deploy the model with:

````sh
truss push --trusted --publish
````


Paste your Baseten API key if prompted. Also ensure the `hf_access_token` secret is properly setup in your Baseten Account to access this model.

**Note**: TensorRT-LLM with engine builder will only work under a Baseten production deployment

For more information, refer to the [Truss documentation](https://docs.baseten.co/performance/engine-builder-overview).