# Whisper Streaming

Deploy Whisper Streaming for speech-to-text transcription on Baseten.

| Property | Value |
|----------|-------|
| Task | Speech-to-text |
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
  -d '{"url": "https://example.com/audio.wav"}'
```

## Configuration highlights

- Base image: `baseten/truss-server-base:3.10-gpu-v0.4.9`
- System packages: `ffmpeg`
