# llava 1.6 SGL

Deploy llava 1.6 SGL for text generation using a Custom (Truss) engine on Baseten.

| Property | Value |
|----------|-------|
| Task | Text generation |
| Engine | Custom (Truss) |
| GPU | A100 |
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
  -d '{"prompt": "What is machine learning?", "max_tokens": 512}'
```

## Configuration highlights

- Predict concurrency: **128**
