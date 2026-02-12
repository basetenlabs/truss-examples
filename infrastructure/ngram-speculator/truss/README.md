# ngram-speculator

Deploy ngram-speculator using a custom server configuration on Baseten.

| Property | Value |
|----------|-------|
| Task | Infrastructure / Custom server |
| Engine | Custom (Truss) |
| GPU | H100 |
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
  -d '{"messages": [{"role": "user", "content": "What is the capital of France?"}], "max_tokens": 256}'
```

## Configuration highlights

- Engine: **Custom (Truss)**
