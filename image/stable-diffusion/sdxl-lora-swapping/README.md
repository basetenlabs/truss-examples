# Stable Diffusion XL with LoRA Swapping

Deploy Stable Diffusion XL with LoRA Swapping for image generation on Baseten.

| Property | Value |
|----------|-------|
| Task | Image generation |
| Engine | Custom (Truss) |
| GPU | A100 |
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
  -d '{
  "lora": {
    "repo_id": "nerijs/pixel-art-xl",
    "weights": "pixel-art-xl.safetensors"
  },
  "prompt": "pixel art, an baby giraffe"
}'
```

## Configuration highlights

- System packages: `ffmpeg, libsm6, libxext6`
