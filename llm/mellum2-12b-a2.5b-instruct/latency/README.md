# Mellum2-12B-A2.5B-Instruct with vLLM — Latency Template

Mellum2-12B-A2.5B-Instruct is JetBrains' open MoE code model with a 131k-token context window. It is optimized for code generation and completion tasks and supports tool calling via the Hermes parser. This template serves the model on a single H100 via the official `vllm/vllm-openai:v0.23.0` image.

> **Note:** vLLM v0.23.0 or later includes `MellumForCausalLM` support; this template pins `vllm/vllm-openai:v0.23.0`.

---

## Requirements

- `truss >= 0.10.5`
- Baseten account with H100 GPU access
- Hugging Face access token with read permission for `JetBrains/Mellum2-12B-A2.5B-Instruct`

---

## Key Configuration

| Parameter | Value | Why it matters |
| --- | --- | --- |
| `base_image` | `vllm/vllm-openai:v0.23.0` | Pinned stable release with MellumForCausalLM support |
| `accelerator` | `H100` | Single H100 (80 GB) fits the model comfortably |
| `max-model-len` | `auto` | Automatically set the maximum context window |
| `tensor-parallel-size` | `$GPU_COUNT` | Dynamic GPU count from `nvidia-smi` |
| `enable-prefix-caching` | — | Reuse cached KV blocks for repeated prompts |
| `load-format` | `runai_streamer` | Stream weights efficiently during load |
| `tool-call-parser` | `hermes` | Enables OpenAI-compatible function calling |
| `predict_concurrency` | `128` | High concurrency for throughput-friendly workloads |

---

## Deployment

```sh
git clone https://github.com/basetenlabs/model-registry.git
cd model-registry/llm/mellum2-12b-a2.5b-instruct/latency
```

Before deploying:

1. Create a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Add your Hugging Face token as a secret named `hf_access_token` in Baseten.
3. Install the latest Truss: `pip install --upgrade truss`

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
    model="JetBrains/Mellum2-12B-A2.5B-Instruct",
    messages=[
        {"role": "user", "content": "Write a Python function to reverse a string."}
    ],
    stream=True,
    max_tokens=4096,
    temperature=0.6,
    top_p=0.95,
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
            "name": "run_python",
            "description": "Execute a Python snippet and return stdout.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to run"}
                },
                "required": ["code"],
            },
        },
    }
]

response = client.chat.completions.create(
    model="JetBrains/Mellum2-12B-A2.5B-Instruct",
    messages=[{"role": "user", "content": "Use run_python to print the first 5 Fibonacci numbers."}],
    tools=tools,
    tool_choice="auto",
)

tool_call = response.choices[0].message.tool_calls[0]
print(tool_call.function.name, json.loads(tool_call.function.arguments))
```

---

## Support

Open an issue in this repository or contact the Baseten support team.