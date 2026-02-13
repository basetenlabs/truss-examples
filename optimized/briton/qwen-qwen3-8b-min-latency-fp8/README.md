# Briton-qwen-qwen3-8b-min-latency-fp8-truss-example

Deploy [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) for text generation using a TRT-LLM engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) |
| Task | Text generation |
| Engine | TRT-LLM |
| GPU | H100 |
| Quantization | FP8 KV |
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
    model="Qwen/Qwen3-8B",
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
  -d '{"model": "Qwen/Qwen3-8B", "messages": [{"role": "user", "content": "What is machine learning?"}], "max_tokens": 512}'
```

## Configuration highlights

- Quantization: **fp8_kv**
- Speculative decoding: **LOOKAHEAD_DECODING**
- Max sequence length: **40,960**
- Chunked context: **enabled**
- Plugin: **use_fp8_context_fmha**
- Streaming: **enabled**
