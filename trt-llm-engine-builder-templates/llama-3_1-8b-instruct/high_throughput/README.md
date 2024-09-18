# Llama 3.1 8B Instruct using TensorRT-LLM with High Throughput

This directory is a base model [Truss](https://truss.baseten.co/) example of the model Llama 3.1 8b using our TensorRT-LLM (TRTLLM) [engine builder](https://docs.baseten.co/performance/engine-builder-overview), catered towards high throughput use cases.

This includes products that prioritize large data processing with medium to low latency such as:
* Content moderation for social media platforms
* Article summarization
* RAG Models


This particular example uses a large `max_batch_size` of `32` and `fp8_kv` quantization to allow for greater throughput. Below is a view of the important configuration values:

| Property             | Value  |
|----------------------|--------|
| GPU                  | 1xH100 |
| `max_batch_size`     |   32   |
| `quantization_type`  |`fp8_kv`|
| `max_input_len`      |  4096  |
| `max_output_len`     |  1024  |


### Metrics
A small benchmarking test was ran on this configuration, conducting 150 requests at 32 concurrent requests at a time, with full load (~4000 input tokens). Important details are below:

| Metric                             | Value      |
|------------------------------------|------------|
| Total Requests                     | 150        |
| Average Latency                    | 1.6489     |
| Average TTFT                       | 1.1599     |
| Average Perceived Tokens per Second| 479.8651   |
| Average Overall Throughput         | 15355.6843 |

## Deployment

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd trt-llm-engine-builder-templates/llama-3_1-8b-instruct/high_throughput
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `trt-llm-engine-builder-templates/llama-3_1-8b-instruct/high_throughput` as your working directory, you can deploy the model with:

```sh
truss push --trusted --publish
```

Paste your Baseten API key if prompted. Also ensure the `hf_access_token` secret is properly setup in your Baseten Account to access this model.

_**Note**: TensorRT-LLM with engine builder will only work under a Baseten production deployment_

For more information, see [Truss documentation](https://docs.baseten.co/performance/engine-builder-overview).
