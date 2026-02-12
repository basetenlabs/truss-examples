# Llama 3.1 Nemotron Nano VL 8B V1

Deploy [nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1) for text generation using a vLLM engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1) |
| Task | Text generation |
| Engine | vLLM |
| GPU | H100:1 |
| OpenAI compatible | Yes |
| Python | py312 |

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
    model="nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1",
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
  -d '{"model": "nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1", "messages": [{"role": "user", "content": "What is machine learning?"}], "max_tokens": 512}'
```

## Configuration highlights

- Base image: `vllm/vllm-openai:v0.11.0`
- Model cache: **volume-mounted** for fast cold starts
- Predict concurrency: **16**
- Streaming: **enabled**
