# TRT Whisper - Dynamic Batching

A tutorial example showing how to deploy TRT Whisper - Dynamic Batching on Baseten.

| Property | Value |
|----------|-------|
| Model | [baseten/trtllm-whisper-a10g-large-v2-1](https://huggingface.co/baseten/trtllm-whisper-a10g-large-v2-1) |
| Task | Tutorial |
| Engine | Custom (Truss) |
| GPU | A10G |
| Python | py311 |

## Deploy

```sh
truss push
```

## Invoke

```sh
curl -X POST https://model-<model_id>.api.baseten.co/predict \
  -H "Authorization: Api-Key YOUR_BASETEN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"audio": "<base64_encoded_wav_audio>"}'
```

## Configuration highlights

- Base image: `baseten/trtllm-server:r23.12_baseten_v0.9.0.dev2024022000`
- Model cache: **volume-mounted** for fast cold starts
- Predict concurrency: **256**
- System packages: `python3.10-venv, ffmpeg`
