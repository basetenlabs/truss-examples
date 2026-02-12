# sesame-csm-1b

Deploy sesame-csm-1b for text-to-speech on Baseten.

| Property | Value |
|----------|-------|
| Task | Text-to-speech |
| Engine | Custom (Truss) |
| GPU | T4 |
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
  "text": "Hello from Sesame.",
  "speaker": 0
}'
```

## Configuration highlights

- Engine: **Custom (Truss)**
