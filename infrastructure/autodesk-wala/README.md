# ADSKAILab/WaLa-SV-1B

Deploy ADSKAILab/WaLa-SV-1B using a custom server configuration on Baseten.

| Property | Value |
|----------|-------|
| Task | Infrastructure / Custom server |
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
  -d '{"image_b64": "<base64_encoded_image>"}'
```

## Configuration highlights

- Engine: **Custom (Truss)**
