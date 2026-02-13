# Qwen3 Reranker 4B

Deploy [michaelfeil/Qwen3-Reranker-4B-seq](https://huggingface.co/michaelfeil/Qwen3-Reranker-4B-seq) as a reranker using a BEI (TensorRT) engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [michaelfeil/Qwen3-Reranker-4B-seq](https://huggingface.co/michaelfeil/Qwen3-Reranker-4B-seq) |
| Task | Reranking |
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
curl -X POST https://model-<model_id>.api.baseten.co/rerank \
  -H "Authorization: Api-Key YOUR_BASETEN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is deep learning?", "texts": ["Deep learning is a subset of machine learning.", "The weather is nice today."], "raw_scores": true}'
```

## Configuration highlights

- Quantization: **fp8**
