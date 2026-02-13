# SDXL ControlNet Canny

Deploy SDXL ControlNet Canny for image generation on Baseten.

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
  "prompt": "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"
}'
```

## Configuration highlights

- System packages: `ffmpeg, libsm6, libxext6`
