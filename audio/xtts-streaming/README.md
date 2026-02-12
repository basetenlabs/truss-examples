# XTTS Streaming - High Performance

Deploy XTTS Streaming - High Performance for text-to-speech on Baseten.

| Property | Value |
|----------|-------|
| Task | Text-to-speech |
| Engine | Custom (Truss) |
| GPU | H100 |

## Deploy

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

- Base image: `htrivedi05/xtts-streaming`
- Environment variables: `COQUI_TOS_AGREED`
