# Llama with Cached Weights

A tutorial example showing how to deploy Llama with Cached Weights on Baseten.

| Property | Value |
|----------|-------|
| Model | [NousResearch/Llama-2-7b-chat-hf](https://huggingface.co/NousResearch/Llama-2-7b-chat-hf) |
| Task | Tutorial |
| Engine | Custom (Truss) |
| GPU | A10G |
| Python | py39 |

## Deploy

```sh
truss push
```

## Invoke

```sh
curl -X POST https://model-<model_id>.api.baseten.co/predict \
  -H "Authorization: Api-Key YOUR_BASETEN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "prompt": "What is the meaning of life?"
}'
```

## Configuration highlights

- Model cache: **volume-mounted** for fast cold starts
