# Alibaba-NLP GTE Qwen2 7B Instruct Embedding

Deploy [Alibaba-NLP/gte-Qwen2-7B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct) for generating text embeddings using a TEI (HuggingFace) engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [Alibaba-NLP/gte-Qwen2-7B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct) |
| Task | Embeddings |
| Engine | TEI (HuggingFace) |
| GPU | H100 |
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
  -d '{"input": "What is deep learning?", "model": "Alibaba-NLP/gte-Qwen2-7B-instruct"}'
```

## Configuration highlights

- Base image: `baseten/text-embeddings-inference-mirror:hopper-1.8.3`
- Model cache: **volume-mounted** for fast cold starts
- Predict concurrency: **32**
