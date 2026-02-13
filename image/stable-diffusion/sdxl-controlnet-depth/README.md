# SDXL ControlNet Depth

Deploy SDXL ControlNet Depth for image generation on Baseten.

| Property | Value |
|----------|-------|
| Task | Image generation |
| Engine | Custom (Truss) |
| GPU | A10G:2 |
| Python | py39 |

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
  "prompt": "large bed, abstract painting on the wall, fluffy rug on the floor, ambient lighting, extremely detailed"
}'
```

## Configuration highlights

- System packages: `ffmpeg, libsm6, libxext6`
