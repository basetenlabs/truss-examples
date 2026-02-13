# Qwen3 Embedding 4B FP4

Deploy [michaelfeil/Qwen3-Embedding-4B-auto](https://huggingface.co/michaelfeil/Qwen3-Embedding-4B-auto) for generating text embeddings using a BEI (TensorRT) engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [michaelfeil/Qwen3-Embedding-4B-auto](https://huggingface.co/michaelfeil/Qwen3-Embedding-4B-auto) |
| Task | Embeddings |
| Engine | BEI (TensorRT) |
| GPU | B200 |
| Quantization | FP4 |
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
  -d '{"input": "What is deep learning?", "model": "michaelfeil/Qwen3-Embedding-4B-auto"}'
```

## Configuration highlights

- Quantization: **fp4**
