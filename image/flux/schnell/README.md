# Flux.1-schnell

Deploy [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) for image generation on Baseten.

| Property | Value |
|----------|-------|
| Model | [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) |
| Task | Image generation |
| Engine | Custom (Truss) |
| GPU | H100_40GB |
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
  "prompt": "black forest gateau cake spelling out the words \"FLUX SCHNELL\", tasty, food photography, dynamic shot"
}'
```

## Configuration highlights

- System packages: `ffmpeg, libsm6, libxext6`
