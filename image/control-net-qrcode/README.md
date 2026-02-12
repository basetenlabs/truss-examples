# control-net-qrcode

Deploy control-net-qrcode for image generation on Baseten.

| Property | Value |
|----------|-------|
| Task | Image generation |
| Engine | Custom (Truss) |
| GPU | T4 |
| Python | py310 |

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
  "prompt": "A cubism painting of the Garden of Eaden with animals walking around, Andreas Rocha, matte painting concept art, a detailed matte painting",
  "qr_code_content": "https://www.baseten.co"
}'
```

## Configuration highlights

- Engine: **Custom (Truss)**
