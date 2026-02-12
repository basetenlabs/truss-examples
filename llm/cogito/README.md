# Cogito v2 Preview Models

Deploy [Deep Cogito](https://www.deepcogito.com/research/cogito-v2-preview) v2 preview models using vLLM's OpenAI-compatible server. These models feature tool calling and reasoning capabilities built on Llama and DeepSeek architectures.

| Variant | Size | Architecture | GPU | Path |
|---------|------|--------------|-----|------|
| Cogito v2 Llama 70B | 70B | Dense | 2x H100 | [`cogito-v2-preview-llama-70B-vllm/`](cogito-v2-preview-llama-70B-vllm/) |
| Cogito v2 Llama 109B MoE | 109B | MoE | 4x H100 | [`cogito-v2-preview-llama-109B-MoE-vllm/`](cogito-v2-preview-llama-109B-MoE-vllm/) |
| Cogito v2 Llama 405B | 405B | Dense | 8x B200 | [`cogito-v2-preview-llama-405B-vllm/`](cogito-v2-preview-llama-405B-vllm/) |
| Cogito v2 DeepSeek 671B MoE | 671B | MoE | 8x B200 | [`cogito-v2-preview-deepseek-671B-MoE-vllm/`](cogito-v2-preview-deepseek-671B-MoE-vllm/) |

## Deploy

> **Note:** These models require a HuggingFace access token. Set `hf_access_token` in your Baseten secrets before deploying.

```sh
truss push llm/cogito/cogito-v2-preview-llama-70B-vllm
```

## Invoke

All models use the OpenAI ChatCompletion format:

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_BASETEN_API_KEY",
    base_url="https://model-<model_id>.api.baseten.co/environments/production/sync/v1",
)

response = client.chat.completions.create(
    model="deepcogito/cogito-v2-preview-llama-70B",
    messages=[{"role": "user", "content": "What is today's temperature in celsius? I'm in Paris."}],
    max_tokens=1000,
)

print(response.choices[0].message.content)
```

```sh
curl -X POST https://model-<model_id>.api.baseten.co/v1/chat/completions \
  -H "Authorization: Api-Key YOUR_BASETEN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "deepcogito/cogito-v2-preview-llama-70B", "messages": [{"role": "user", "content": "What is machine learning?"}], "max_tokens": 512}'
```
