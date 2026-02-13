# Alibaba-NLP GTE Qwen2 1.5B Instruct Embedding

Deploy [Alibaba-NLP/gte-Qwen2-1.5B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct) for generating text embeddings using a TEI (HuggingFace) engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [Alibaba-NLP/gte-Qwen2-1.5B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct) |
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
  -d '{"input": "What is deep learning?", "model": "Alibaba-NLP/gte-Qwen2-1.5B-instruct"}'
```

## Configuration highlights

- Base image: `baseten/text-embeddings-inference-mirror:89-1.8.3`
- Model cache: **volume-mounted** for fast cold starts
- Predict concurrency: **32**
