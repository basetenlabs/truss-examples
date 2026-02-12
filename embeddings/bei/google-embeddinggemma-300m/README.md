# Google EmbeddingGemma 300M Embedding

Deploy [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m) for generating text embeddings using a BEI (TensorRT) engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m) |
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
  -d '{"input": "What is deep learning?", "model": "google/embeddinggemma-300m"}'
```

## Configuration highlights

- Engine: **BEI (TensorRT)**
