# Falcon 3 10B Instruct using TensorRT-LLM with Speculative Decoding

This directory is [Truss](https://truss.baseten.co/) template for deploying model Llama 3.1 8B Instruct using our TensorRT-LLM (TRT-LLM) [engine builder](https://docs.baseten.co/performance/engine-builder-overview). This configuration is optimized for high-throughput scenarios.

## Use case

This deployment is tailored for applications that require processing large volumes of data with moderate to low latency, such as:
* Content moderation for social media platforms
* Bulk article summarization
* Large-scale Retrieval-Augmented Generation (RAG) systems

## Configuration

The template uses the following key configuration parameters:

| Property            | Value    | Description                                                                    |
| ------------------- | -------- | ------------------------------------------------------------------------------ |
| GPU                 | 1xH100   | Single NVIDIA H100 GPU                                                         |
| `max_batch_size`    | 32       | Allows processing up to 32 requests simultaneously                             |
| `quantization_type` | `fp8_kv` | FP8 quantization, balancing performance and accuracy. See this [blog](https://www.baseten.co/blog/33-faster-llm-inference-with-fp8-quantization/). |
| `max_input_len`     | 4096     | Maximum number of input tokens that the model will accept                      |
| `max_output_len`    | 1024     | Maximum number of output tokens the model can generate                         |


## Deployment

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd falcon/falcon3-10B-trt-llm-spec-dec
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `falcon/falcon3-10B-trt-llm-spec-dec` as your working directory, you can deploy the model with:

```sh
truss push --trusted --publish
```

Paste your Baseten API key if prompted. Also ensure the `hf_access_token` secret is properly setup in your Baseten Account to access this model.

**Note**: TensorRT-LLM with engine builder will only work under a Baseten production deployment

For more information, refer to the [Truss documentation](https://docs.baseten.co/performance/engine-builder-overview).
