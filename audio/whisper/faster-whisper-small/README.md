# Faster Whisper Small

A small speech-to-text model for multi-lingual audio transcription.

| Property | Value |
|----------|-------|
| Model | [Systran/faster-whisper-small](https://huggingface.co/Systran/faster-whisper-small) |
| Task | Speech-to-text |
| Engine | Custom (Truss) |
| GPU | T4 |
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
