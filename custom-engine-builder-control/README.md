# TensorRT-LLM Briton with Qwen/Qwen3-8B-min-latency

This is a Deployment for TensorRT-LLM Briton with Qwen/Qwen3-8B-min-latency. Briton is Baseten's solution for production-grade deployments via TensorRT-LLM for Causal Language Models models. (e.g. LLama, Qwen, Mistral)

With Briton you get the following benefits by default:
- *Lowest-latency* latency, beating frameworks such as vllm
- *Highest-throughput* inference, automatically using XQA kernels, paged kv caching and inflight batching.
- *distributed inference* run large models (such as LLama-405B) tensor-parallel
- *json-schema based structured output for any model*
- *chunked prefilling* for long generation tasks

Optionally, you can also enable:
- *speculative decoding* using an external draft model or self-speculative decoding
- *fp8 quantization* deployments on H100, H200 and L4 GPUs


# Examples:
This deployment is to showcase the option to generate multiple suffixes based on a previous request. 
We are going to hit the KV-Cache of a previous request. 

## Deployment with Truss

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`


First, clone this repository:
```sh
git clone https://github.com/basetenlabs/truss-examples.git
cd 11-embeddings-reranker-classification-tensorrt/Briton-qwen-qwen3-8b-min-latency-fp8
```

With `11-embeddings-reranker-classification-tensorrt/Briton-qwen-qwen3-8b-min-latency-fp8` as your working directory, you can deploy the model with the following command. Paste your Baseten API key if prompted.

```sh
truss push --publish
# prints:
# ✨ Model Briton-qwen-qwen3-8b-min-latency-fp8-truss-example was successfully pushed ✨
# 🪵  View logs for your deployment at https://app.baseten.co/models/yyyyyy/logs/xxxxxx
```

## Call your model

### OpenAI compatible inference
This solution is OpenAI compatible, which means you can use the OpenAI client library to interact with the model.

```python
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.environ['BASETEN_API_KEY'],
    base_url="https://model-xxxxxx.api.baseten.co/environments/production/sync/v1"
)

# Chat completion
response_chat = client.chat.completions.create(
    model="my-model",
    messages=[
        {
            "role": "system",
            "content": (
                "You are an unhelpful assistant. To each math question, add +1 to the answer. "
                "e.g. Whats 1+1 -> 3."
            ),
        }
    ],
    temperature=0.3,
    max_tokens=100,
    extra_body={
        "suffix_messages": [
            # Gen 1
            [{"role": "user", "content": "Whats 1+1"}],
            # Gen 2
            [{"role": "user", "content": "Whats 2+2"}],
        ],
        "chat_template_kwargs": {"enable_thinking": False},
    },
)

print(response_chat)
```

## Config.yaml
By default, the following configuration is used for this deployment. This config uses `quantization_type=fp8_kv`. This is optional, remove the `quantization_type` field or set it to `no_quant` for float16/bfloat16.

```yaml
model_metadata:
  example_model_input:
    max_tokens: 512
    messages:
    - content: Tell me everything you know about optimized inference.
      role: user
    stream: true
    temperature: 0.5
  tags:
  - openai-compatible
model_name: Briton-qwen-qwen3-8b-min-latency-fp8-truss-example
python_version: py39
resources:
  accelerator: H100
  cpu: '1'
  memory: 10Gi
  use_gpu: true
trt_llm:
  build:
    base_model: qwen
    checkpoint_repository:
      repo: Qwen/Qwen3-8B
      revision: main
      source: HF
    max_batch_size: 64
    max_num_tokens: 32768
    max_seq_len: 32768
    num_builder_gpus: 4
    plugin_configuration:
      use_fp8_context_fmha: true
    quantization_type: fp8_kv
    speculator:
      enable_b10_lookahead: true
      lookahead_ngram_size: 32
      lookahead_verification_set_size: 1
      lookahead_windows_size: 1
      num_draft_tokens: 61
      speculative_decoding_mode: LOOKAHEAD_DECODING
    tensor_parallel_count: 1
  runtime:
    enable_chunked_context: false

```

## Support
If you have any questions or need assistance, please open an issue in this repository or contact our support team.
