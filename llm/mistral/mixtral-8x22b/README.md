# Mixtral 8x22

Deploy [mistralai/Mixtral-8x22B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1) for text generation using a Custom (Truss) engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [mistralai/Mixtral-8x22B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1) |
| Task | Text generation |
| Engine | Custom (Truss) |
| GPU | A100:4 |
| Python | py310 |

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
  "prompt": "What is the Mistral wind?"
}'
```

## Configuration highlights

- Engine: **Custom (Truss)**
