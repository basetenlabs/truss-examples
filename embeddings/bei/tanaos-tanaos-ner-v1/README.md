# Tanaos NER v1

Deploy [tanaos/tanaos-NER-v1](https://huggingface.co/tanaos/tanaos-NER-v1) for classification using a BEI (TensorRT) engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [tanaos/tanaos-NER-v1](https://huggingface.co/tanaos/tanaos-NER-v1) |
| Task | Classification |
| Engine | BEI (TensorRT) |
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

- Engine: **BEI (TensorRT)**
