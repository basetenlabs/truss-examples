# Pixtral 12B

Deploy [mistral-community/pixtral-12b-240910](https://huggingface.co/mistral-community/pixtral-12b-240910) for text generation using a Custom (Truss) engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [mistral-community/pixtral-12b-240910](https://huggingface.co/mistral-community/pixtral-12b-240910) |
| Task | Text generation |
| Engine | Custom (Truss) |
| GPU | A100 |
| Python | py311 |

## Deploy

> **Note:** This model requires a HuggingFace access token. Set `hf_access_token` in your Baseten secrets before deploying.

```sh
truss push
```

## Invoke

```sh
curl -X POST https://model-<model_id>.api.baseten.co/predict \
  -H "Authorization: Api-Key YOUR_BASETEN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
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

- Engine: **Custom (Truss)**
