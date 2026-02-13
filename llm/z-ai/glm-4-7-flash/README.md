# GLM 4.7 Flash

Deploy [zai-org/GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash) for text generation using a SGLang engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [zai-org/GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash) |
| Task | Text generation |
| Engine | SGLang |
| GPU | H100:2 |
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
    model="zai-org/GLM-4.7-Flash",
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
  -d '{"model": "zai-org/GLM-4.7-Flash", "messages": [{"role": "user", "content": "What is machine learning?"}], "max_tokens": 512}'
```

## Configuration highlights

- Base image: `lmsysorg/sglang:nightly-dev-20260122-e6ccb294`
- Predict concurrency: **32**
- Streaming: **enabled**
