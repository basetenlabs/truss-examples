# Alibaba-NLP GTE Reranker ModernBERT Base

Deploy [Alibaba-NLP/gte-reranker-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base) as a reranker using a TEI (HuggingFace) engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [Alibaba-NLP/gte-reranker-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base) |
| Task | Reranking |
| Engine | TEI (HuggingFace) |
| GPU | L4 |
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

- Base image: `baseten/text-embeddings-inference-mirror:89-1.8.3`
- Model cache: **volume-mounted** for fast cold starts
- Predict concurrency: **32**
