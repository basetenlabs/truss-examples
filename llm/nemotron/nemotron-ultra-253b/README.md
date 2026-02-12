# Briton-nemotron-253b-tp8-fp8

Deploy [michaelfeil/nemotron-251b-ultra-v2-tp8-fp8-tllm](https://huggingface.co/michaelfeil/nemotron-251b-ultra-v2-tp8-fp8-tllm) for text generation using a TRT-LLM engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [michaelfeil/nemotron-251b-ultra-v2-tp8-fp8-tllm](https://huggingface.co/michaelfeil/nemotron-251b-ultra-v2-tp8-fp8-tllm) |
| Task | Text generation |
| Engine | TRT-LLM |
| GPU | H100:8 |
| Quantization | FP8 |
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
    model="michaelfeil/nemotron-251b-ultra-v2-tp8-fp8-tllm",
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
  -d '{"model": "michaelfeil/nemotron-251b-ultra-v2-tp8-fp8-tllm", "messages": [{"role": "user", "content": "What is machine learning?"}], "max_tokens": 512}'
```

## Configuration highlights

- Quantization: **fp8**
- Tensor parallelism: **8** GPUs
- Speculative decoding: **LOOKAHEAD_DECODING**
- Max sequence length: **65,536**
- Streaming: **enabled**
