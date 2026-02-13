# Chatterbox TTS

Deploy Chatterbox TTS for text-to-speech on Baseten.

| Property | Value |
|----------|-------|
| Task | Text-to-speech |
| Engine | Custom (Truss) |
| GPU | H100 |
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
  -d '{"text": "Hello, this is a test of text to speech."}'
```

## Configuration highlights

- Base image: `jojobaseten/truss-numpy-1.26.0-gpu:0.4`
