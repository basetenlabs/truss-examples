# Briton-google-gemma-3-1b-it-truss-example

Deploy [unsloth/gemma-3-1b-it](https://huggingface.co/unsloth/gemma-3-1b-it) for text generation using a TRT-LLM engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [unsloth/gemma-3-1b-it](https://huggingface.co/unsloth/gemma-3-1b-it) |
| Task | Text generation |
| Engine | TRT-LLM |
| GPU | H100_40GB |
| Quantization | NO QUANT |
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
    model="unsloth/gemma-3-1b-it",
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
  -d '{"model": "unsloth/gemma-3-1b-it", "messages": [{"role": "user", "content": "What is machine learning?"}], "max_tokens": 512}'
```

## Configuration highlights

- Quantization: **no_quant**
- Max sequence length: **32,768**
- Chunked context: **enabled**
- Batch scheduler policy: **max_utilization**
- Streaming: **enabled**
