# Llama 3.1 8B Instruct using TensorRT-LLM for Large Context

This directory is a [Truss](https://truss.baseten.co/) template for deploying the Llama 3.1 8B Instruct model using our TensorRT-LLM (TRT-LLM) [engine builder](https://docs.baseten.co/performance/engine-builder-overview). This configuration is optimized for large context scenarios.

## Use case

This deployment is tailored for applications that require processing extensive input with high accuracy, such as:
* Textbook summarization
* Codebase analysis
* Multi-turn conversation

## Configuration

The template uses the following key configuration parameters:

| Property            | Value  | Description                                               |
| ------------------- | ------ | --------------------------------------------------------- |
| GPU                 | 1xH100 | Single NVIDIA H100 GPU                                    |
| `max_batch_size`    | 16     | Allows processing up to 16 requests simultaneously        |
| `quantization_type` | `fp16` | Maintain FP16 to retain model accuracy                    |
| `max_input_len`     | 8192   | Maximum number of input tokens that the model will accept |
| `max_output_len`    | 4096   | Maximum number of output tokens the model can generate    |

## Performance Metrics
A preliminary benchmark was conducted with the following parameters:
- 150 total requests
- 16 concurrent requests
- ~8000 input tokens per request

Results:

| Metric                              | Value               |
| ----------------------------------- | ------------------- |
| Average Latency                     | 1.8100 s            |
| Average Time to First Token (TTFT)  | 1.1407 s            |
| Average Perceived Tokens per Second | 812.0655            |
| Average Overall Throughput          | 12993.0482 tokens/s |

## Deployment

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd trt-llm-engine-builder-templates/llama-3_1-8b-instruct/large_context
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `trt-llm-engine-builder-templates/llama-3_1-8b-instruct/large_context` as your working directory, you can deploy the model with:

```sh
truss push --trusted --publish
```

Paste your Baseten API key if prompted. Also ensure the `hf_access_token` secret is properly setup in your Baseten Account to access this model.

**Note**: TensorRT-LLM with engine builder will only work under a Baseten production deployment

For more information, refer to the [Truss documentation](https://docs.baseten.co/performance/engine-builder-overview).
