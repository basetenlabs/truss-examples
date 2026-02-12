# bert

A tutorial example showing how to deploy bert on Baseten.

| Property | Value |
|----------|-------|
| Task | Tutorial |
| Engine | Custom (Truss) |
| GPU | CPU |
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
  "text": "Hello my name is {MASK}"
}'
```

## Configuration highlights

- Engine: **Custom (Truss)**
