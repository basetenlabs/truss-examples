# Stable Diffusion XL

A tutorial example showing how to deploy Stable Diffusion XL on Baseten.

| Property | Value |
|----------|-------|
| Task | Tutorial |
| Engine | Custom (Truss) |
| GPU | A10G |
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
  "prompt": "A tree in a field under the night sky",
  "use_refiner": true
}'
```

## Configuration highlights

- System packages: `ffmpeg, libsm6, libxext6`
