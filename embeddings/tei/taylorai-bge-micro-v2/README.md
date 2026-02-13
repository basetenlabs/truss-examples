# TaylorAI BGE Micro v2

Deploy [TaylorAI/bge-micro-v2](https://huggingface.co/TaylorAI/bge-micro-v2) for generating text embeddings using a TEI (HuggingFace) engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [TaylorAI/bge-micro-v2](https://huggingface.co/TaylorAI/bge-micro-v2) |
| Task | Embeddings |
| Engine | TEI (HuggingFace) |
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

- Base image: `baseten/text-embeddings-inference-mirror:86-1.8.3`
- Model cache: **volume-mounted** for fast cold starts
- Predict concurrency: **32**
