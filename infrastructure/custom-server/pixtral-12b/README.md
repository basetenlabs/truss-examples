# Pixtral 12B

Deploy [mistralai/Pixtral-12B-2409](https://huggingface.co/mistralai/Pixtral-12B-2409) using a custom server configuration on Baseten.

| Property | Value |
|----------|-------|
| Model | [mistralai/Pixtral-12B-2409](https://huggingface.co/mistralai/Pixtral-12B-2409) |
| Task | Infrastructure / Custom server |
| Engine | vLLM |
| GPU | H100 |
| OpenAI compatible | Yes |

## Deploy

```sh
truss push
```

## Invoke

```sh
curl -X POST https://model-<model_id>.api.baseten.co/v1/chat/completions \
  -H "Authorization: Api-Key YOUR_BASETEN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "model": "pixtral",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Describe this image in one sentence."
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "https://picsum.photos/id/237/200/300"
          }
        }
      ]
    }
  ],
  "stream": false,
  "max_tokens": 512,
  "temperature": 0.5
}'
```

## Configuration highlights

- Base image: `vllm/vllm-openai:v0.7.3`
- Predict concurrency: **16**
- Environment variables: `VLLM_LOGGING_LEVEL`
