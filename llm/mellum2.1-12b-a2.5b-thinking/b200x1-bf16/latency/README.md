# Mellum2.1 12B A2.5B Thinking (BF16), 1x B200, BIS-LLM

[JetBrains/Mellum2.1-12B-A2.5B-Thinking-MTP](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking-MTP)
on a single B200 with BIS-LLM (vLLM engine backend behind Dynamo cache-aware routing), served as
`JetBrains/Mellum2.1-12B-A2.5B-Thinking-MTP` behind an OpenAI-compatible `/v1/chat/completions`.

Mellum2.1 Thinking is JetBrains' 12B mixture-of-experts reasoning model (2.5B active parameters)
for agentic coding: working in a repository, running commands, and calling tools.

- Full 131,072-token native context, prefix caching, chunked prefill.
- MTP speculative decoding with the checkpoint's own multi-token prediction head, 2 draft tokens.
  The draft only speeds up decoding; every token is verified, so outputs are unchanged.
- Thinking is on by default. Reasoning is returned in `reasoning_content`, separate from the
  answer. Turn it off per request with `chat_template_kwargs: {"enable_thinking": false}`.
- Hermes-format tool calling and `response_format` JSON schemas.
- Text only.

JetBrains recommends `temperature=0.6`, `top_p=0.95` and `top_k=20` for this model.

## Deploy

```sh
git clone https://github.com/basetenlabs/model-registry.git
cd model-registry/llm/mellum2.1-12b-a2.5b-thinking/b200x1-bf16/latency
baseten model push
```

Install the Baseten CLI with `brew tap basetenlabs/baseten && brew install baseten`. The weights
are public (Apache 2.0); no Hugging Face token is needed.

## Call the model

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["BASETEN_API_KEY"],
    base_url="https://model-xxxxxx.api.baseten.co/environments/production/sync/v1",
)

response = client.chat.completions.create(
    model="JetBrains/Mellum2.1-12B-A2.5B-Thinking-MTP",
    messages=[
        {
            "role": "user",
            "content": "Find the bug in this function and explain the fix: "
            "def mean(xs): return sum(xs) / len(xs) - 1",
        }
    ],
    max_tokens=8192,
    temperature=0.6,
    top_p=0.95,
    extra_body={"top_k": 20},
)
message = response.choices[0].message
print(getattr(message, "reasoning_content", None))  # the model's reasoning
print(message.content)  # the answer
```

### Tool calling

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run the test suite and return the output.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Test file or directory"}},
                "required": ["path"],
            },
        },
    }
]

response = client.chat.completions.create(
    model="JetBrains/Mellum2.1-12B-A2.5B-Thinking-MTP",
    messages=[{"role": "user", "content": "Run the tests in tests/test_stats.py."}],
    tools=tools,
    tool_choice="auto",
    max_tokens=8192,
)
print(response.choices[0].message.tool_calls)
```