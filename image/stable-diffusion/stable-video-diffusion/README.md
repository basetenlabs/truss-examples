# Stable Video Diffusion

Stable Video Diffusion can turn any image into a short video.

| Property | Value |
|----------|-------|
| Task | Image generation |
| Engine | Custom (Truss) |
| GPU | A100 |
| Python | py310 |

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

- System packages: `libgl1-mesa-glx, ffmpeg`
