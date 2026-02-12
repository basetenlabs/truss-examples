# DeepSeek V2.5 1210 SGLang

Deploy [deepseek-ai/DeepSeek-V2.5-1210](https://huggingface.co/deepseek-ai/DeepSeek-V2.5-1210) using a custom server configuration on Baseten.

| Property | Value |
|----------|-------|
| Model | [deepseek-ai/DeepSeek-V2.5-1210](https://huggingface.co/deepseek-ai/DeepSeek-V2.5-1210) |
| Task | Infrastructure / Custom server |
| Engine | SGLang |
| GPU | H100:8 |

## Deploy

> **Note:** This model requires a HuggingFace access token. Set `hf_access_token` in your Baseten secrets before deploying.

```sh
truss push
```

## Invoke

```sh
curl -X POST https://model-<model_id>.api.baseten.co/v1/completions \
  -H "Authorization: Api-Key YOUR_BASETEN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek-ai/DeepSeek-V2.5-1210", "prompt": "What is the capital of France?", "max_tokens": 256}'
```

## Configuration highlights

- Base image: `lmsysorg/sglang:v0.4.0.post1-cu124`
- Predict concurrency: **32**
