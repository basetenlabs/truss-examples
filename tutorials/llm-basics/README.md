# Mistral 7B

A tutorial example showing how to deploy Mistral 7B on Baseten.

| Property | Value |
|----------|-------|
| Task | Tutorial |
| Engine | Custom (Truss) |
| GPU | A10G |
| Python | py311 |

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
  "prompt": "What is the meaning of life?"
}'
```

## Configuration highlights

- Engine: **Custom (Truss)**
