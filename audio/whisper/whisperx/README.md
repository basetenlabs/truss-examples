# whisperX

Deploy whisperX for speech-to-text transcription on Baseten.

| Property | Value |
|----------|-------|
| Task | Speech-to-text |
| Engine | Custom (Truss) |
| GPU | L4 |
| Python | py310 |

## Deploy

> **Note:** This model requires a HuggingFace access token. Set `hf_access_token` in your Baseten secrets before deploying.

```sh
truss push
```

## Invoke

```sh
curl -X POST https://model-<model_id>.api.baseten.co/predict \
  -H "Authorization: Api-Key YOUR_BASETEN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "audio_file": "https://cdn.baseten.co/docs/production/Gettysburg.mp3"
}'
```

## Configuration highlights

- Base image: `runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04`
- System packages: `ffmpeg, libsm6, libxext6`
