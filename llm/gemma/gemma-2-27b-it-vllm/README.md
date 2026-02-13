# Gemma 2 27B Instruct VLLM

Deploy [google/gemma-2-27b-it](https://huggingface.co/google/gemma-2-27b-it) for text generation using a Custom (Truss) engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [google/gemma-2-27b-it](https://huggingface.co/google/gemma-2-27b-it) |
| Task | Text generation |
| Engine | Custom (Truss) |
| GPU | A100 |
| Python | py311 |

## Deploy

> **Note:** This model requires a HuggingFace access token. Set `hf_access_token` in your Baseten secrets before deploying.

```sh
truss push
```

## Invoke

```sh
curl -X POST https://model-<model_id>.api.baseten.co/predict \
  -H "Authorization: Api-Key YOUR_BASETEN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "prompt": "what is the meaning of life"
}'
```

## Configuration highlights

- Predict concurrency: **128**
