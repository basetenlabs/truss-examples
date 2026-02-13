# Nomic AI Nomic Embed Text v1.5

Deploy [nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) for generating text embeddings using a BEI (TensorRT) engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) |
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
  -d '{"input": "What is deep learning?", "model": "nomic-ai/nomic-embed-text-v1.5"}'
```

## Configuration highlights

- Engine: **BEI (TensorRT)**
