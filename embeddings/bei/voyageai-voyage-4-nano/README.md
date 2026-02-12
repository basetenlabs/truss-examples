# VoyageAI Voyage 4 Nano

Deploy [voyageai/voyage-4-nano](https://huggingface.co/voyageai/voyage-4-nano) for generating text embeddings using a BEI (TensorRT) engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [voyageai/voyage-4-nano](https://huggingface.co/voyageai/voyage-4-nano) |
| Task | Embeddings |
| Engine | BEI (TensorRT) |
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
  -d '{"input": "What is deep learning?", "model": "voyageai/voyage-4-nano"}'
```

## Configuration highlights

- Engine: **BEI (TensorRT)**
