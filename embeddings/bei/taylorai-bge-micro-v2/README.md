# TaylorAI BGE Micro v2

Deploy [TaylorAI/bge-micro-v2](https://huggingface.co/TaylorAI/bge-micro-v2) for generating text embeddings using a BEI (TensorRT) engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [TaylorAI/bge-micro-v2](https://huggingface.co/TaylorAI/bge-micro-v2) |
| Task | Embeddings |
| Engine | BEI (TensorRT) |
| GPU | A10G |
| Python | py39 |

## Deploy

```sh
truss push
```

## Invoke

```sh
curl -X POST https://model-<model_id>.api.baseten.co/v1/embeddings \
  -H "Authorization: Api-Key YOUR_BASETEN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": "What is deep learning?", "model": "TaylorAI/bge-micro-v2"}'
```

## Configuration highlights

- Engine: **BEI (TensorRT)**
