# Llama 3.1 70B Instruct using TensorRT-LLM with High Throughput

This directory is a base model [Truss](https://truss.baseten.co/) example of the model Llama 3.1 70b using our TensorRT-LLM (TRTLLM) [engine builder](https://docs.baseten.co/performance/engine-builder-overview), catered towards high throughput use cases.

This includes products that prioritize large data processing with medium to low latency such as:
* Content moderation for social media platforms
* Article summarization
* RAG Models


This particular example uses a `max_batch_size` of `16` wrt this large model, and `fp8_kv` quantization to allow for greater throughput. Below is a view of the important configuration values:

| Metric               | Value  |
|----------------------|--------|
| GPU                  | 2xH100 |
| `max_batch_size`     |   16   |
| `quantization_type`  |`fp8_kv`|
| `max_input_len`      |  4096  |
| `max_output_len`     |  1024  |


### Metrics
TODO

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

_**Note**: TensorRT-LLM with engine builder will only work under a Baseten production deployment_

For more information, see [Truss documentation](https://docs.baseten.co/performance/engine-builder-overview).
