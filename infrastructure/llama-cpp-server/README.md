# llama cpp gemma 3 27b it qat q4_0

Deploy [google/gemma-3-27b-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-3-27b-it-qat-q4_0-gguf) using a custom server configuration on Baseten.

| Property | Value |
|----------|-------|
| Model | [google/gemma-3-27b-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-3-27b-it-qat-q4_0-gguf) |
| Task | Infrastructure / Custom server |
| Engine | Docker Server |
| GPU | H100 |
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
  "model": "gemma",
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
  "stream": true,
  "max_tokens": 512,
  "temperature": 0.5
}'
```

## Configuration highlights

- Base image: `alphatozeta/llama-cpp-server:0.4`
- Predict concurrency: **8**
- Streaming: **enabled**
