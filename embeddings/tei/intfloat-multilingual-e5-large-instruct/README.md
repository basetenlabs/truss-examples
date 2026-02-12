# Intfloat Multilingual E5 Large Instruct Embedding

Deploy [intfloat/multilingual-e5-large-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct) for generating text embeddings using a TEI (HuggingFace) engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [intfloat/multilingual-e5-large-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct) |
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
  -d '{"input": "What is deep learning?", "model": "intfloat/multilingual-e5-large-instruct"}'
```

## Configuration highlights

- Base image: `baseten/text-embeddings-inference-mirror:89-1.8.3`
- Model cache: **volume-mounted** for fast cold starts
- Predict concurrency: **32**
