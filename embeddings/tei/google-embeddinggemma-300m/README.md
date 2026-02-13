# Google EmbeddingGemma 300M Embedding

Deploy [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m) for generating text embeddings using a TEI (HuggingFace) engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m) |
| Task | Embeddings |
| Engine | TEI (HuggingFace) |
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

- Base image: `baseten/text-embeddings-inference-mirror:89-1.8.3`
- Model cache: **volume-mounted** for fast cold starts
- Predict concurrency: **32**
