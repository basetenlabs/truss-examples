# Briton-meta-llama-llama-3.1-405b-fp8-truss-example

Deploy [https://mp-model-weights-public.s3.us-east-2.amazonaws.com/llama-405b-tp8-fp8kv-tllm.tar](https://huggingface.co/https://mp-model-weights-public.s3.us-east-2.amazonaws.com/llama-405b-tp8-fp8kv-tllm.tar) for text generation using a TRT-LLM engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [https://mp-model-weights-public.s3.us-east-2.amazonaws.com/llama-405b-tp8-fp8kv-tllm.tar](https://huggingface.co/https://mp-model-weights-public.s3.us-east-2.amazonaws.com/llama-405b-tp8-fp8kv-tllm.tar) |
| Task | Text generation |
| Engine | TRT-LLM |
| GPU | H100:8 |
| Quantization | FP8 KV |
| OpenAI compatible | Yes |
| Python | py39 |

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
    model="https://mp-model-weights-public.s3.us-east-2.amazonaws.com/llama-405b-tp8-fp8kv-tllm.tar",
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
  -d '{"model": "https://mp-model-weights-public.s3.us-east-2.amazonaws.com/llama-405b-tp8-fp8kv-tllm.tar", "messages": [{"role": "user", "content": "What is machine learning?"}], "max_tokens": 512}'
```

## Configuration highlights

- Quantization: **fp8_kv**
- Tensor parallelism: **8** GPUs
- Max sequence length: **131,072**
- Chunked context: **enabled**
- Plugin: **use_fp8_context_fmha**
- Streaming: **enabled**
- Environment variables: `ENABLE_EXECUTOR_API`
