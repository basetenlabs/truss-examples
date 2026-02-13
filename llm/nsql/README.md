# NSQL 350M

NSQL is an open-source text-to-SQL AI model developed by Numbers Station.

| Property | Value |
|----------|-------|
| Task | Text generation |
| Engine | Custom (Truss) |
| GPU | A10G |
| Python | py39 |

## Deploy

```sh
truss push
```

## Invoke

```sh
curl -X POST https://model-<model_id>.api.baseten.co/predict \
  -H "Authorization: Api-Key YOUR_BASETEN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is machine learning?", "max_tokens": 512}'
```

## Configuration highlights

- Engine: **Custom (Truss)**
