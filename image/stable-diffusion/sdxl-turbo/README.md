# SDXL Turbo

Deploy [stabilityai/sdxl-turbo](https://huggingface.co/stabilityai/sdxl-turbo) for image generation on Baseten.

| Property | Value |
|----------|-------|
| Model | [stabilityai/sdxl-turbo](https://huggingface.co/stabilityai/sdxl-turbo) |
| Task | Image generation |
| Engine | Custom (Truss) |
| GPU | T4 |
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
  -d '{
  "prompt": "A tree in a field under the night sky"
}'
```

## Configuration highlights

- Engine: **Custom (Truss)**
