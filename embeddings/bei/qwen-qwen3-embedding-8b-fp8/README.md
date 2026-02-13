# Qwen3 Embedding 8B

Deploy [michaelfeil/Qwen3-Embedding-8B-auto](https://huggingface.co/michaelfeil/Qwen3-Embedding-8B-auto) for generating text embeddings using a BEI (TensorRT) engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [michaelfeil/Qwen3-Embedding-8B-auto](https://huggingface.co/michaelfeil/Qwen3-Embedding-8B-auto) |
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
  -d '{"input": "What is deep learning?", "model": "michaelfeil/Qwen3-Embedding-8B-auto"}'
```

## Configuration highlights

- Quantization: **fp8**
