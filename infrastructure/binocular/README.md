# Binoculars

Deploy [tiiuae/falcon-7b](https://huggingface.co/tiiuae/falcon-7b) using a custom server configuration on Baseten.

| Property | Value |
|----------|-------|
| Model | [tiiuae/falcon-7b](https://huggingface.co/tiiuae/falcon-7b) |
| Task | Infrastructure / Custom server |
| Engine | Custom (Truss) |
| GPU | A10G:2 |
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
  -d '{"text": "Your text to check for AI-generated content goes here. The input should be at least 64 tokens long for reliable detection."}'
```

## Configuration highlights

- Engine: **Custom (Truss)**
