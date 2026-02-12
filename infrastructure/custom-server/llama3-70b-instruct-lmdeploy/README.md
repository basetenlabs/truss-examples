# Llama 3.1 70B Instruct LMDeploy

Deploy [meta-llama/Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct) using a custom server configuration on Baseten.

| Property | Value |
|----------|-------|
| Model | [meta-llama/Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct) |
| Task | Infrastructure / Custom server |
| Engine | Docker Server |
| GPU | H100:4 |

## Deploy

> **Note:** This model requires a HuggingFace access token. Set `hf_access_token` in your Baseten secrets before deploying.

```sh
truss push
```

## Invoke

```sh
curl -X POST https://model-<model_id>.api.baseten.co/v1/completions \
  -H "Authorization: Api-Key YOUR_BASETEN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.1-70B-Instruct", "prompt": "What is the capital of France?", "max_tokens": 256}'
```

## Configuration highlights

- Base image: `openmmlab/lmdeploy:v0.6.4-cu12`
- Predict concurrency: **32**
