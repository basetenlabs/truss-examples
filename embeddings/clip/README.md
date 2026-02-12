# clip-example

Deploy clip-example for generating text embeddings using a Custom (Truss) engine on Baseten.

| Property | Value |
|----------|-------|
| Task | Embeddings |
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
  -d '{
  "url": "https://images.pexels.com/photos/1170986/pexels-photo-1170986.jpeg?auto=compress&cs=tinysrgb&w=1600"
}'
```

## Configuration highlights

- Engine: **Custom (Truss)**
