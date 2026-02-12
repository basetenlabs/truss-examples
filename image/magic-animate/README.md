# Magic Animate

Deploy Magic Animate for image generation on Baseten.

| Property | Value |
|----------|-------|
| Task | Image generation |
| Engine | Custom (Truss) |
| GPU | A10G |
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
  "guidance_scale": 7.5,
  "motion_sequence": "<BASE64 MP4 FILE>",
  "reference_image": "<BASE64 IMAGE>",
  "seed": 1,
  "steps": 10
}'
```

## Configuration highlights

- System packages: `ffmpeg`
