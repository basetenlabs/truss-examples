# Laguna M.1 with vLLM — Latency Template

Laguna M.1 is Poolside's flagship reasoning model, a Mixture-of-Experts (MoE) architecture optimized for agentic coding and extended reasoning tasks. This FP8 checkpoint (~225 GB) runs on 4× H100 GPUs and is served via the official `vllm/vllm-openai` container with Poolside's custom tool-call and reasoning parsers.

This template is latency-optimized: `--max-num-seqs 64` limits concurrent sequences to minimize head-of-line blocking from long thinking traces, making it well-suited for agentic workloads where individual request latency gates overall task time.

---

## Requirements

- `truss >= 0.10.5`
- Baseten account with H100 GPU access

---

## Key Configuration

| Parameter | Value | Why it matters |
| --- | --- | --- |
| `base_image` | `vllm/vllm-openai:v0.21.0` | Minimum version with Laguna support |
| `accelerator` | `H100:4` | FP8 ~225 GB fits in 4× H100 (320 GB VRAM) with headroom |
| `tensor-parallel-size` | `4` | Shards model across all 4 GPUs |
| `max-num-seqs` | `64` | Caps concurrent sequences to reduce queuing behind long thinking traces |
| `gpu-memory-utilization` | `0.95` | Maximizes available VRAM for KV cache |
| `max-model-len` | `262144` | 256 K context window |
| `tool-call-parser` | `poolside_v1` | Poolside-native tool call format |
| `reasoning-parser` | `poolside_v1` | Enables extended thinking extraction |
| `enable_thinking` | `true` | Default chat template activates reasoning traces |
| `predict_concurrency` | `64` | Matches max-num-seqs at the Truss level |

---

## Deployment

Clone the repository:

```sh
git clone https://github.com/basetenlabs/model-registry.git
cd model-registry/llm/laguna-m.1/latency
```

Before deploying:

1. Create a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest Truss: `pip install --upgrade truss`

Deploy:

```sh
truss push --trusted --publish
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