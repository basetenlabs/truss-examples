# NVIDIA Llama Nemotron Embed 1B v2

Deploy [nvidia/llama-nemotron-embed-1b-v2](https://huggingface.co/nvidia/llama-nemotron-embed-1b-v2) for generating text embeddings using a BEI (TensorRT) engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [nvidia/llama-nemotron-embed-1b-v2](https://huggingface.co/nvidia/llama-nemotron-embed-1b-v2) |
| Task | Embeddings |
| Engine | BEI (TensorRT) |
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
  -d '{"input": "What is deep learning?", "model": "nvidia/llama-nemotron-embed-1b-v2"}'
```

## Configuration highlights

- Engine: **BEI (TensorRT)**
