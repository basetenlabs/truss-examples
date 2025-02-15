# TensorRT-LLM Briton with meta-llama/Llama-3.1-405B

This is a Deployment for TensorRT-LLM Briton with meta-llama/Llama-3.1-405B. Briton is Baseten's solution for production-grade deployments via TensorRT-LLM for Causal Language Models models. (e.g. LLama, Qwen, Mistral)

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
This deployment is specifically designed for the Hugging Face model [meta-llama/Llama-3.1-405B](https://huggingface.co/meta-llama/Llama-3.1-405B).
Suitable models can be identified by the `ForCausalLM` suffix in the model name. Currently we support e.g. LLama, Qwen, Mistral models.

meta-llama/Llama-3.1-405B  is a text-generation model, used to generate text given a prompt. \nIt is frequently used in chatbots, text completion, structured output and more.


## Deployment with Truss

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`
Note: [This is a gated/private model] Retrieve your Hugging Face token from the [settings](https://huggingface.co/settings/tokens). Set your Hugging Face token as a Baseten secret [here](https://app.baseten.co/settings/secrets) with the key `hf_access_token`. Do not set the actual value of key in the config.yaml. `hf_access_token: null` is fine - the true value will be fetched from the secret store.

First, clone this repository:
```sh
git clone https://github.com/basetenlabs/truss-examples.git
cd 11-embeddings-reranker-classification-tensorrt/Briton-meta-llama-llama-3.1-405b
```

With `11-embeddings-reranker-classification-tensorrt/Briton-meta-llama-llama-3.1-405b` as your working directory, you can deploy the model with the following command. Paste your Baseten API key if prompted.

```sh
truss push --publish
# prints:
# ✨ Model Briton-meta-llama-llama-3.1-405b-truss-example was successfully pushed ✨
# 🪵  View logs for your deployment at https://app.baseten.co/models/yyyyyy/logs/xxxxxx
```

## Call your model

### OpenAI compatible inference
Briton is OpenAI compatible, which means you can use the OpenAI client library to interact with the model.

```python
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.environ['BASETEN_API_KEY'],
    base_url="https://model-xxxxxx.api.baseten.co/environments/production/sync/v1"
)

# Default completion
response_completion = client.completions.create(
    model="not_required",
    prompt="Q: Tell me everything about Baseten.co! A:",
    temperature=0.3,
    max_tokens=100,
)

# Chat completion
response_chat = client.chat.completions.create(
    model="",
    messages=[
        {"role": "user", "content": "Tell me everything about Baseten.co!"}
    ],
    temperature=0.3,
    max_tokens=100,
)

# Structured output
from pydantic import BaseModel

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

completion = client.beta.chat.completions.parse(
    model="not_required",
    messages=[
        {"role": "system", "content": "Extract the event information."},
        {"role": "user", "content": "Alice and Bob are going to a science fair on Friday."},
    ],
    response_format=CalendarEvent,
)

event = completion.choices[0].message.parsed

# If you model supports tool-calling, you can use the following example:
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current temperature for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and country e.g. Bogotá, Colombia"
                }
            },
            "required": [
                "location"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
}]

completion = client.chat.completions.create(
    model="not_required",
    messages=[{"role": "user", "content": "What is the weather like in Paris today?"}],
    tools=tools
)

print(completion.choices[0].message.tool_calls)
```


## Config.yaml
By default, the following configuration is used for this deployment. This config uses `quantization_type=fp8_kv`. This is optional, remove the `quantization_type` field or set it to `no_quant` for float16/bfloat16.
Note: [This is a gated/private model] Retrieve your Hugging Face token from the [settings](https://huggingface.co/settings/tokens). Set your Hugging Face token as a Baseten secret [here](https://app.baseten.co/settings/secrets) with the key `hf_access_token`. Do not set the actual value of key in the config.yaml. `hf_access_token: null` is fine - the true value will be fetched from the secret store.
```yaml
build_commands: []
environment_variables: {}
external_package_dirs: []
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
model_name: Briton-meta-llama-llama-3.1-405b-truss-example
python_version: py39
requirements: []
resources:
  accelerator: H100:8
  cpu: '1'
  memory: 10Gi
  use_gpu: true
secrets:
  hf_access_token: null
system_packages: []
trt_llm:
  build:
    base_model: llama
    checkpoint_repository:
      repo: meta-llama/Llama-3.1-405B
      revision: main
      source: HF
    max_seq_len: 131072
    plugin_configuration:
      use_fp8_context_fmha: true
    quantization_type: fp8_kv
    tensor_parallel_count: 8
  runtime:
    enable_chunked_context: true

```

## Support
If you have any questions or need assistance, please open an issue in this repository or contact our support team.
