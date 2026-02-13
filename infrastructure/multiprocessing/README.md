# Model with multiprocessing pre/post-process

Deploy Model with multiprocessing pre/post-process using a custom server configuration on Baseten.

| Property | Value |
|----------|-------|
| Task | Infrastructure / Custom server |
| Engine | Custom (Truss) |
| GPU | A10G |
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
  -d '{"n": 100}'
```

## Configuration highlights

- Engine: **Custom (Truss)**
