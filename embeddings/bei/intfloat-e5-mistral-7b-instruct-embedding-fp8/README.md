# Intfloat E5 Mistral 7B Instruct Embedding

Deploy [intfloat/e5-mistral-7b-instruct](https://huggingface.co/intfloat/e5-mistral-7b-instruct) for generating text embeddings using a BEI (TensorRT) engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [intfloat/e5-mistral-7b-instruct](https://huggingface.co/intfloat/e5-mistral-7b-instruct) |
| Task | Embeddings |
| Engine | BEI (TensorRT) |
| GPU | H100 |
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
  -d '{"input": "What is deep learning?", "model": "intfloat/e5-mistral-7b-instruct"}'
```

## Configuration highlights

- Quantization: **fp8**
