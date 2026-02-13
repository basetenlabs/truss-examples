# Phi-3-Mini-128K-Instruct

Deploy Phi-3-Mini-128K-Instruct for text generation using a Custom (Truss) engine on Baseten.

| Property | Value |
|----------|-------|
| Task | Text generation |
| Engine | Custom (Truss) |
| GPU | T4 |
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
