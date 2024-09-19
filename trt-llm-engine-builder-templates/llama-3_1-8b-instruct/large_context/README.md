# Llama 3.1 8B Instruct using TensorRT-LLM with High Throughput

This directory is a base model [Truss](https://truss.baseten.co/) example of the model Llama 3.1 8b using our TensorRT-LLM (TRTLLM) [engine builder](https://docs.baseten.co/performance/engine-builder-overview), catered towards large context use cases.

This includes products that prioritize large data processing and accuracy over latency.
* Textbook summarization
* Codebase analysis
* Multi-turn conversation

This particular example uses a very large `max_input_len` and `max_output_len`, smaller batch size of `16`, along with standard `no_quant` quantization to retain model accuracy. Below is a view of the important configuration values:

| Property             | Value  |
|----------------------|--------|
| GPU                  | 1xH100 |
| `max_batch_size`     |   16   |
| `quantization_type`  | `fp16` |
| `max_input_len`      |  8192  |
| `max_output_len`     |  4096  |


### Metrics
A small benchmarking test was ran on this configuration, conducting 150 requests at 16 concurrent requests at a time, with full input load (~8000 input tokens). Details are below:

| Metric                             | Value      |
|------------------------------------|------------|
| Total Requests                     | 150        |
| Average Latency                    | 1.8100     |
| Average TTFT                       | 1.1407     |
| Total Prompt Tokens                | 204450     |
| Total Completion Tokens            | 15150      |
| Total Tokens                       | 219600     |
| Average Perceived Tokens per Second| 812.0655   |
| Average Overall Throughput         | 12993.0482 |

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

_**Note**: TensorRT-LLM with engine builder will only work under a Baseten production deployment_

For more information, see [Truss documentation](https://docs.baseten.co/performance/engine-builder-overview).
