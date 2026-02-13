# Gemma 27B Instruct

Deploy [google/gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it) for text generation using a vLLM engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [google/gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it) |
| Task | Text generation |
| Engine | vLLM |
| GPU | H100 |
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
    model="google/gemma-3-27b-it",
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
  -d '{"model": "google/gemma-3-27b-it", "messages": [{"role": "user", "content": "What is machine learning?"}], "max_tokens": 512}'
```

## Configuration highlights

- Base image: `public.ecr.aws/q9t5s3a7/vllm-ci-postmerge-repo:8a4a2efc6fc32cdc30e4e35ba3f8c64dcd0aa1d0`
- Model cache: **volume-mounted** for fast cold starts
- Predict concurrency: **8**
- Streaming: **enabled**
- Environment variables: `VLLM_LOGGING_LEVEL`
