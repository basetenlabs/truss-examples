# BISV2-nvidia-llama-3.1-8b-instruct-fp4-truss-example

Deploy [nvidia/Llama-3.1-8B-Instruct-FP4](https://huggingface.co/nvidia/Llama-3.1-8B-Instruct-FP4) for text generation using a TRT-LLM engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [nvidia/Llama-3.1-8B-Instruct-FP4](https://huggingface.co/nvidia/Llama-3.1-8B-Instruct-FP4) |
| Task | Text generation |
| Engine | TRT-LLM |
| GPU | B200 |
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
    model="nvidia/Llama-3.1-8B-Instruct-FP4",
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
  -d '{"model": "nvidia/Llama-3.1-8B-Instruct-FP4", "messages": [{"role": "user", "content": "What is machine learning?"}], "max_tokens": 512}'
```

## Configuration highlights

- Quantization: **no_quant**
- Streaming: **enabled**
