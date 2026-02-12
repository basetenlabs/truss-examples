# Mixtral 8x7B Instruct TRT-LLM Weights Only Quantized

Mixtral 8x7B Instruct, with INT8 weights only quantization, optimized with TRT-LLM! Compatible with OpenAI Client

| Property | Value |
|----------|-------|
| Model | [mistralai/Mixtral-8x7B-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) |
| Task | Text generation |
| Engine | Custom (Truss) |
| GPU | A100 |
| OpenAI compatible | Yes |
| Python | py311 |

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
    model="mistralai/Mixtral-8x7B-v0.1",
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
  -d '{"model": "mistralai/Mixtral-8x7B-v0.1", "messages": [{"role": "user", "content": "What is machine learning?"}], "max_tokens": 512}'
```

## Configuration highlights

- Base image: `docker.io/baseten/triton_trt_llm:main-20231215`
- Predict concurrency: **256**
