# GPT OSS 20B

Deploy [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) for text generation using a TRT-LLM engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) |
| Task | Text generation |
| Engine | TRT-LLM |
| GPU | H100 |
| OpenAI compatible | Yes |

## Deploy

```sh
truss push
```

## Invoke

This model is OpenAI-compatible. You can use the OpenAI Python client or curl.

**Python (OpenAI SDK):**

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_BASETEN_API_KEY",
    base_url="https://model-<model_id>.api.baseten.co/v1",
)

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[{"role": "user", "content": "What is machine learning?"}],
    max_tokens=512,
)

print(response.choices[0].message.content)
```

**curl:**

```sh
curl -X POST https://model-<model_id>.api.baseten.co/v1/chat/completions \
  -H "Authorization: Api-Key YOUR_BASETEN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "openai/gpt-oss-20b", "messages": [{"role": "user", "content": "What is machine learning?"}], "max_tokens": 512}'
```

## Configuration highlights

- Model cache: **volume-mounted** for fast cold starts
- Streaming: **enabled**
