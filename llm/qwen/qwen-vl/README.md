# Qwen VL

Deploy [Qwen/Qwen-VL](https://huggingface.co/Qwen/Qwen-VL) for text generation using a Custom (Truss) engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [Qwen/Qwen-VL](https://huggingface.co/Qwen/Qwen-VL) |
| Task | Text generation |
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
  -d '{"prompt": "What is machine learning?", "max_tokens": 512}'
```

## Configuration highlights

- Engine: **Custom (Truss)**
