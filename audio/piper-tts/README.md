# Piper TTS

Deploy Piper TTS for text-to-speech on Baseten.

| Property | Value |
|----------|-------|
| Task | Text-to-speech |
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
  -d '{
  "text": "I love robots. Robots are cool!"
}'
```

## Configuration highlights

- Engine: **Custom (Truss)**
