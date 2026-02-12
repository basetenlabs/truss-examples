# Whisper V3

Transcribe audio files across multiple languages.

| Property | Value |
|----------|-------|
| Task | Speech-to-text |
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
  "url": "https://cdn.baseten.co/docs/production/Gettysburg.mp3"
}'
```

## Configuration highlights

- System packages: `ffmpeg`
