# AudioGen medium

AudioGen is a simple and controllable model for audio generation developed by Facebook AI Research.

| Property | Value |
|----------|-------|
| Task | Audio generation |
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
  "duration": 8,
  "prompts": [
    "dog barking",
    "sirene of an emergency vehicle",
    "footsteps in a corridor"
  ]
}'
```

## Configuration highlights

- System packages: `ffmpeg`
