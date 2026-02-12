# BAAI BGE M3 Embedding Dense

Deploy [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) for generating text embeddings using a BEI (TensorRT) engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) |
| Task | Embeddings |
| Engine | BEI (TensorRT) |
| GPU | H100 |
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
  -d '{"input": "What is deep learning?", "model": "BAAI/bge-m3"}'
```

## Configuration highlights

- Engine: **BEI (TensorRT)**
