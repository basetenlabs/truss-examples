# ngram-speculator

Deploy [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) using a custom server configuration on Baseten.

| Property | Value |
|----------|-------|
| Model | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) |
| Task | Infrastructure / Custom server |
| Engine | vLLM |
| GPU | H100 |

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
  "model": "llama",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What do llamas dream of?"
        }
      ]
    }
  ],
  "stream": false,
  "max_tokens": 512
}'
```

## Configuration highlights

- Base image: `vllm/vllm-openai:nightly`
- Predict concurrency: **16**
