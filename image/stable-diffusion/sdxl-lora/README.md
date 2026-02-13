# Stable Diffusion XL with LoRA

Deploy Stable Diffusion XL with LoRA for image generation on Baseten.

| Property | Value |
|----------|-------|
| Task | Image generation |
| Engine | Custom (Truss) |
| GPU | A10G |
| Python | py311 |

## Deploy

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

- System packages: `ffmpeg, libsm6, libxext6`
