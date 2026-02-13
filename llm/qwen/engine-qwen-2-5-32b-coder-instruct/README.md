# Qwen Coder 2.5 32B Instruct

Deploy [Qwen/Qwen2.5-Coder-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct) for text generation using a TRT-LLM engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [Qwen/Qwen2.5-Coder-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct) |
| Task | Text generation |
| Engine | TRT-LLM |
| GPU | H100 |
| Quantization | FP8 |
| OpenAI compatible | Yes |
| Python | py39 |

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
    model="Qwen/Qwen2.5-Coder-32B-Instruct",
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
  -d '{"model": "Qwen/Qwen2.5-Coder-32B-Instruct", "messages": [{"role": "user", "content": "What is machine learning?"}], "max_tokens": 512}'
```

## Configuration highlights

- Quantization: **fp8**
- Max sequence length: **32,768**
- Chunked context: **enabled**
- Batch scheduler policy: **max_utilization**
- Plugin: **use_paged_context_fmha**
- Plugin: **paged_kv_cache**
- Streaming: **enabled**
