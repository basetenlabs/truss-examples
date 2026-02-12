# Cogito V2 Preview Llama 405B vLLM

Deploy [deepcogito/cogito-v2-preview-llama-405B](https://huggingface.co/deepcogito/cogito-v2-preview-llama-405B) for text generation using a vLLM engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [deepcogito/cogito-v2-preview-llama-405B](https://huggingface.co/deepcogito/cogito-v2-preview-llama-405B) |
| Task | Text generation |
| Engine | vLLM |
| GPU | B200:8 |
| OpenAI compatible | Yes |

## Deploy

> **Note:** This model requires a HuggingFace access token. Set `hf_access_token` in your Baseten secrets before deploying.

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
    model="deepcogito/cogito-v2-preview-llama-405B",
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
  -d '{"model": "deepcogito/cogito-v2-preview-llama-405B", "messages": [{"role": "user", "content": "What is machine learning?"}], "max_tokens": 512}'
```

## Configuration highlights

- Base image: `vllm/vllm-openai:v0.10.0`
- Predict concurrency: **32**
- Streaming: **enabled**
- Environment variables: `VLLM_LOGGING_LEVEL`
