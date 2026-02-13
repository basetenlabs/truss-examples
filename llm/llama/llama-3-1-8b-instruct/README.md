# Llama 3.1 8B Instruct VLLM

Deploy [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) for text generation using a Custom (Truss) engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| Task | Text generation |
| Engine | Custom (Truss) |
| GPU | H100_40GB |
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
