# Sentence Transformers All-MiniLM-L6-v2 Embedding

Deploy [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) for generating text embeddings using a TEI (HuggingFace) engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| Task | Embeddings |
| Engine | TEI (HuggingFace) |
| GPU | T4 |
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
  -d '{"input": "What is deep learning?", "model": "sentence-transformers/all-MiniLM-L6-v2"}'
```

## Configuration highlights

- Base image: `baseten/text-embeddings-inference-mirror:turing-1.8.3`
- Model cache: **volume-mounted** for fast cold starts
- Predict concurrency: **32**
