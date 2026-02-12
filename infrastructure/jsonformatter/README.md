# JsonFormatter

Deploy JsonFormatter using a custom server configuration on Baseten.

| Property | Value |
|----------|-------|
| Task | Infrastructure / Custom server |
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
  -d '{"prompt": "Generate a person'\''s name and age"}'
```

## Configuration highlights

- Engine: **Custom (Truss)**
