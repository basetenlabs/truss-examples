# Baseten Inference Stack with nvidia/Qwen3-30B-A3B-FP4

This is a Deployment for Baseten Inference Stack with nvidia/Qwen3-30B-A3B-FP4. Baseten Inference Stack is Baseten's solution for production-grade deployments via TensorRT-LLM for Causal Language Models models. (e.g. LLama, Qwen, Mistral)

With Baseten Inference Stack you get the following benefits by default:
- *Lowest-latency* latency, beating frameworks such as vllm
- *Highest-throughput* inference, automatically using XQA kernels, paged kv caching and inflight batching.
- *distributed inference* run large models (such as LLama-405B) tensor-parallel
- *json-schema based structured output for any model*
- *chunked prefilling* for long generation tasks

Optionally, you can also enable:
- *speculative decoding* using an external draft model or self-speculative decoding
- *fp8 quantization* deployments on H100, H200 and L4 GPUs
- *fp4 quantization* deployments on B200 GPUs to get even more speed


# Examples:
This deployment is specifically designed for the Hugging Face model [nvidia/Qwen3-30B-A3B-FP4](https://huggingface.co/nvidia/Qwen3-30B-A3B-FP4).
Suitable models can be identified by the `ForCausalLM` suffix in the model name. Currently we support e.g. LLama, Qwen, Mistral models.

nvidia/Qwen3-30B-A3B-FP4  is a text-generation model, used to generate text given a prompt. \nIt is frequently used in chatbots, text completion, structured output and more.


## Deployment with Truss

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`


First, clone this repository:
```sh
git clone https://github.com/basetenlabs/truss-examples.git
cd 11-embeddings-reranker-classification-tensorrt/BISV2-nvidia-qwen3-30b-a3b-fp4
```

With `11-embeddings-reranker-classification-tensorrt/BISV2-nvidia-qwen3-30b-a3b-fp4` as your working directory, you can deploy the model with the following command. Paste your Baseten API key if prompted.

```sh
truss push --publish
# prints:
# ✨ Model BISV2-nvidia-qwen3-30b-a3b-fp4-truss-example was successfully pushed ✨
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
By default, the following configuration is used for this deployment.

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
model_name: BISV2-nvidia-qwen3-30b-a3b-fp4-truss-example
python_version: py39
resources:
  accelerator: B200
  cpu: '1'
  memory: 10Gi
  use_gpu: true
trt_llm:
  build:
    checkpoint_repository:
      repo: nvidia/Qwen3-30B-A3B-FP4
      revision: main
      source: HF
    quantization_type: no_quant
  runtime:
    max_batch_size: 32
    max_num_tokens: 32768
    max_seq_len: 32768

```

## Support
If you have any questions or need assistance, please open an issue in this repository or contact our support team.
