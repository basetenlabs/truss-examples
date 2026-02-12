# Jina AI Jina Code Embeddings 0.5B

Deploy [jinaai/jina-code-embeddings-0.5b](https://huggingface.co/jinaai/jina-code-embeddings-0.5b) for generating text embeddings using a BEI (TensorRT) engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [jinaai/jina-code-embeddings-0.5b](https://huggingface.co/jinaai/jina-code-embeddings-0.5b) |
| Task | Embeddings |
| Engine | BEI (TensorRT) |
| GPU | H100_40GB |
| Quantization | FP8 |
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
  -d '{"input": "What is deep learning?", "model": "jinaai/jina-code-embeddings-0.5b"}'
```

## Configuration highlights

- Quantization: **fp8**
