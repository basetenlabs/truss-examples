# Llama 3.1 70B Instruct using TensorRT-LLM with Low TTFT

This directory is a base model [Truss](https://truss.baseten.co/) example of the model Llama 3.1 70b using our TensorRT-LLM (TRTLLM) [engine builder](https://docs.baseten.co/performance/engine-builder-overview), catered towards low TTFT (Time to First Token).

This includes products that prioritize low latency to the first token generated. Some use cases include:
* Real time chat assistants
* Voice assistance
* Code editor auto-complete
* Translation services

This particular example uses prefix caching, small `max_batch_size` of `4`, and `fp8_kv` quantization to reduce our TTFT. 
Below is a view of the important configuration values:


| Property             | Value  |
|----------------------|--------|
| GPU                  | 2xH100 |
| `max_batch_size`     |   4    |
| `quantization_type`  |`fp8_kv`|
| `max_input_len`      |  4096  |
| `max_output_len`     |  1024  |
| `prefix_caching`     | `true` |


### Metrics

A small benchmarking test was ran on this configuration, conducting 150 requests at 8 concurrent requests at a time, with full input load (~4000 input tokens). Important details are below:

| Metric                             | Value      |
|------------------------------------|------------|
| Total Requests                     | 150        |
| Average Latency                    | 1.3686     |
| Average TTFT                       | 0.7580     |
| Average Perceived Tokens per Second| 530.0073   |
| Average Overall Throughput         | 2120.0290  |

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

```sh
truss push --trusted --publish
```

Paste your Baseten API key if prompted. Also ensure the `hf_access_token` secret is properly setup in your Baseten Account to access this model.

_**Note**: TensorRT-LLM with engine builder will only work under a Baseten production deployment_

For more information, see [Truss documentation](https://docs.baseten.co/performance/engine-builder-overview).
