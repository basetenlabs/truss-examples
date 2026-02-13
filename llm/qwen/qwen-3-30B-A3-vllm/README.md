# Qwen 3 30B-A3B vLLM

Deploy [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) for text generation using a vLLM engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| Task | Text generation |
| Engine | vLLM |
| GPU | H100:1 |
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
    model="Qwen/Qwen3-30B-A3B",
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
  -d '{"model": "Qwen/Qwen3-30B-A3B", "messages": [{"role": "user", "content": "What is machine learning?"}], "max_tokens": 512}'
```

## Configuration highlights

- Base image: `vllm/vllm-openai:v0.8.5`
- Predict concurrency: **32**
- Environment variables: `VLLM_LOGGING_LEVEL`
