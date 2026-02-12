# Qwen3 Omni 30B Instruct

Deploy Qwen3 Omni 30B Instruct on Baseten using a Custom (Truss) engine.

| Property | Value |
|----------|-------|
| Task | Audio |
| Engine | Custom (Truss) |
| GPU | H100 |

## Deploy

```sh
truss push
```

## Invoke

```sh
curl -X POST https://model-<model_id>.api.baseten.co/predict \
  -H "Authorization: Api-Key YOUR_BASETEN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "speaker": "Chelsie",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Hi, how are you?"
        }
      ]
    }
  ]
}'
```

## Configuration highlights

- Base image: `qwenllm/qwen3-omni`
- Predict concurrency: **1**
- Environment variables: `VLLM_LOGGING_LEVEL`
