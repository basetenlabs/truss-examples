# Seed-OSS-36B-Instruct

Deploy [ByteDance-Seed/Seed-OSS-36B-Instruct](https://huggingface.co/ByteDance-Seed/Seed-OSS-36B-Instruct) for text generation using a vLLM engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [ByteDance-Seed/Seed-OSS-36B-Instruct](https://huggingface.co/ByteDance-Seed/Seed-OSS-36B-Instruct) |
| Task | Text generation |
| Engine | vLLM |
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
    model="ByteDance-Seed/Seed-OSS-36B-Instruct",
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
  -d '{"model": "ByteDance-Seed/Seed-OSS-36B-Instruct", "messages": [{"role": "user", "content": "What is machine learning?"}], "max_tokens": 512}'
```

## Configuration highlights

- Base image: `public.ecr.aws/q9t5s3a7/vllm-ci-postmerge-repo:78336a0c3ee4eb9dba6e37959d926160e91623fd`
- Model cache: **volume-mounted** for fast cold starts
- Predict concurrency: **128**
- Streaming: **enabled**
