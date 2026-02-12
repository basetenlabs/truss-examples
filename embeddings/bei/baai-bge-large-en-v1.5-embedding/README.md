# BAAI BGE Large EN v1.5 Embedding

Deploy [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) for generating text embeddings using a BEI (TensorRT) engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) |
| Task | Embeddings |
| Engine | BEI (TensorRT) |
| GPU | L4 |
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
  -d '{"input": "What is deep learning?", "model": "BAAI/bge-large-en-v1.5"}'
```

## Configuration highlights

- Engine: **BEI (TensorRT)**
