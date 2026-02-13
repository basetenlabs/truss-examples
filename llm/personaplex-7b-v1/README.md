# Personaplex 7V H100

Speech-to-speech model powered by NVIDIA Personaplex

| Property | Value |
|----------|-------|
| Model | [nvidia/personaplex-7b-v1](https://huggingface.co/nvidia/personaplex-7b-v1) |
| Task | Text generation |
| Engine | Docker Server |
| GPU | H100 |

## Deploy

> **Note:** This model requires a HuggingFace access token. Set `hf_access_token` in your Baseten secrets before deploying.

```sh
truss push
```

## Invoke

```sh
curl -X POST https://model-<model_id>.api.baseten.co/api/chat \
  -H "Authorization: Api-Key YOUR_BASETEN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is machine learning?", "max_tokens": 512}'
```

## Configuration highlights

- Base image: `basetenservice/personaplex-7v:fork`
