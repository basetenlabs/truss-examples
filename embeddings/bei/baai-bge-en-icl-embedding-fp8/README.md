# BAAI BGE EN ICL Embedding

Deploy [BAAI/bge-en-icl](https://huggingface.co/BAAI/bge-en-icl) for generating text embeddings using a BEI (TensorRT) engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [BAAI/bge-en-icl](https://huggingface.co/BAAI/bge-en-icl) |
| Task | Embeddings |
| Engine | BEI (TensorRT) |
| GPU | H100 |
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
  -d '{"input": "What is deep learning?", "model": "BAAI/bge-en-icl"}'
```

## Configuration highlights

- Quantization: **fp8**
