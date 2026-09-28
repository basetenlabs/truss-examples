# Laguna M.1 with TensorRT-LLM — Latency Template

Laguna M.1 is Poolside's flagship reasoning model, a Mixture-of-Experts (MoE) architecture optimized for agentic coding and extended reasoning tasks. This template serves the NVFP4 checkpoint on 4× B200 GPUs with the Baseten Inference Stack (TensorRT-LLM engine, KV-cache-aware routing), using Poolside's Laguna tool-call and reasoning parsers.

---

## Requirements

- Baseten account with B200 GPU access

---

## Key Configuration

| Parameter | Value | Why it matters |
| --- | --- | --- |
| `checkpoint_name` | `poolside/Laguna-M.1-NVFP4` | NVFP4 weights for Blackwell FP4 kernels |
| `instance_type` | `B200:4` | Four B200 GPUs |
| `tensor_parallel_size` | `4` | Shards the model across all 4 GPUs |
| `max_batch_size` | `128` | Concurrent sequences per replica |
| `max_num_tokens` | `16384` | Tokens scheduled per engine step, with chunked prefill |
| `max_seq_len` | `262144` | 256 K context window |
| `kv_cache_config` | FP8, block reuse, 300 GB host cache | Prefix reuse across turns of agentic sessions |
| `tool_call_parser` / `reasoning_parser` | `laguna` | Poolside-native tool calls and thinking extraction |
| `default_thinking_enabled` | `true` | Reasoning traces on by default |
| `b10_routing_config` | KV-cache-aware routing | Sends requests to the replica holding their prefix |

---

## Deployment

Clone the repository:

```sh
git clone https://github.com/basetenlabs/model-registry.git
cd model-registry/llm/laguna-m.1/latency
```

Before deploying:

1. Create a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the Baseten CLI: `brew tap basetenlabs/baseten && brew install baseten`

Deploy:

```sh
baseten model push
```

---

## Call your model

### Streaming chat completion

```python
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.environ["BASETEN_API_KEY"],
    base_url="https://model-xxxxxx.api.baseten.co/environments/production/sync/v1",
)

response = client.chat.completions.create(
    model="poolside/laguna-m.1",
    messages=[
        {"role": "user", "content": "Write a Python retry wrapper with exponential backoff."}
    ],
    stream=True,
    temperature=1.0,
    top_k=20,
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Tool calling

```python
import json
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.environ["BASETEN_API_KEY"],
    base_url="https://model-xxxxxx.api.baseten.co/environments/production/sync/v1",
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City and country"}
                },
                "required": ["location"],
            },
        },
    }
]

response = client.chat.completions.create(
    model="poolside/laguna-m.1",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools,
    tool_choice="auto",
)

tool_call = response.choices[0].message.tool_calls[0]
print(tool_call.function.name, json.loads(tool_call.function.arguments))
```

---

## Support

Open an issue in this repository or contact our support team.