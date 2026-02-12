# LLM with Streaming

A tutorial example showing how to deploy LLM with Streaming on Baseten.

| Property | Value |
|----------|-------|
| Task | Tutorial |
| Engine | Custom (Truss) |
| GPU | A10G |

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
  "prompt": "what is the meaning of life"
}'
```

## Configuration highlights

- Engine: **Custom (Truss)**
