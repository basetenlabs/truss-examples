# Deepfloyd XL

Generate original images from text prompts.

| Property | Value |
|----------|-------|
| Task | Image generation |
| Engine | Custom (Truss) |
| GPU | A10G |
| Python | py39 |

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
  -d '{"prompt": "A photo of a cat in a field of sunflowers"}'
```

> The response may contain base64-encoded image data.

## Configuration highlights

- Engine: **Custom (Truss)**
