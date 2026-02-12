# ZenCtrl

Deploy ZenCtrl for image generation on Baseten.

| Property | Value |
|----------|-------|
| Task | Image generation |
| Engine | Docker Server |
| GPU | H100 |

## Deploy

> **Note:** This model requires a HuggingFace access token. Set `hf_access_token` in your Baseten secrets before deploying.

```sh
truss push
```

## Invoke

```sh
curl -X POST https://model-<model_id>.api.baseten.co/generate \
  -H "Authorization: Api-Key YOUR_BASETEN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A photo of a cat in a field of sunflowers"}'
```

> The response may contain base64-encoded image data.

## Configuration highlights

- Base image: `fotographerai/zenctrlstage:latest`
- Predict concurrency: **8**
- Environment variables: `PORT`
