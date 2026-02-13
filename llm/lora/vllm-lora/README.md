# Mistral-7B-Instruct VLLM Lora

Deploy [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) for text generation using a vLLM engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) |
| Task | Text generation |
| Engine | vLLM |
| GPU | H100_40GB |

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
  "model": "finance",
  "messages": [
    {
      "role": "user",
      "content": "How would you choose back in 2008?"
    }
  ],
  "stream": true,
  "max_tokens": 512,
  "temperature": 0.9
}'
```

## Configuration highlights

- Base image: `vllm/vllm-openai:v0.9.2`
- Predict concurrency: **32**
- Streaming: **enabled**
