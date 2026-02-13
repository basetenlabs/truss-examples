# private-model

A tutorial example showing how to deploy private-model on Baseten.

| Property | Value |
|----------|-------|
| Task | Tutorial |
| Engine | Custom (Truss) |
| GPU | CPU |
| Python | py39 |

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
  -d '"It is a [MASK] world"'
```

## Configuration highlights

- Engine: **Custom (Truss)**
