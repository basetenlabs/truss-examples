# Mistral-7B-Instruct SGLang Lora

Deploy [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) for text generation using a SGLang engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) |
| Task | Text generation |
| Engine | SGLang |
| GPU | H100_40GB |

## Deploy

> **Note:** This model requires a HuggingFace access token. Set `hf_access_token` in your Baseten secrets before deploying.

```sh
truss push
```

## Invoke

```sh
curl -X POST https://model-<model_id>.api.baseten.co/generate \
  -H "Authorization: Api-Key YOUR_BASETEN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "text": [
    "What would you choose in 2008?",
    "What would you choose in 2008?"
  ],
  "sampling_params": {
    "max_new_tokens": 1000,
    "temperature": 1.0
  },
  "lora_path": [
    "legal",
    "finance"
  ]
}'
```

## Configuration highlights

- Base image: `lmsysorg/sglang:v0.4.9.post6-cu126`
- Predict concurrency: **32**
