# truss_fastapi_datadog

Deploy [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) using a custom server configuration on Baseten.

| Property | Value |
|----------|-------|
| Model | [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| Task | Infrastructure / Custom server |
| Engine | vLLM |
| GPU | H100:1 |
| OpenAI compatible | Yes |

## Deploy

> **Note:** This model requires a HuggingFace access token. Set `hf_access_token` in your Baseten secrets before deploying.

```sh
truss push
```

## Invoke

```sh
curl -X POST https://model-<model_id>.api.baseten.co/v1/chat/completions \
  -H "Authorization: Api-Key YOUR_BASETEN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "What does Tongyi Qianwen mean?"
    }
  ],
  "stream": false,
  "model": "qwen30b",
  "max_tokens": 512,
  "temperature": 0.7
}'
```

## Configuration highlights

- Base image: `chriswirick/truss_fastapi_datadog_vllm:v0.11.0h`
- Predict concurrency: **32**
- Environment variables: `DD_SITE`, `DD_HOSTNAME`, `DD_SERVICE`, `DD_ENV`, `DD_RUN_PATH`, `DD_AUTH_TOKEN_FILE_PATH`, `DD_INVENTORIES_CHECKS_ENABLED`, `DD_OTLP_CONFIG_RECEIVER_PROTOCOLS_GRPC_ENDPOINT`, `DD_CLOUD_PROVIDER_METADATA`, `VLLM_LOGGING_LEVEL`
