# Faster Whisper v3

Faster Whisper v3

| Property | Value |
|----------|-------|
| Model | [Systran/faster-whisper-large-v3](https://huggingface.co/Systran/faster-whisper-large-v3) |
| Task | Speech-to-text |
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
  "url": "https://cdn.baseten.co/docs/production/Gettysburg.mp3"
}'
```

## Configuration highlights

- Engine: **Custom (Truss)**
