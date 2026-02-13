# Parakeet TDT 0.6B V2

Parakeet TDT 0.6B V2 is a 600-million-parameter automatic speech recognition (ASR) model designed for high-quality English transcription.

| Property | Value |
|----------|-------|
| Model | [nvidia/parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) |
| Task | Audio |
| Engine | Custom (Truss) |
| GPU | H100_40GB |
| Python | py312 |

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
  "audio_url": "https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav",
  "timestamps": false
}'
```

## Configuration highlights

- Predict concurrency: **8**
- System packages: `ffmpeg`
