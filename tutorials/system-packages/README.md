# LayoutLM Document QA

A tutorial example showing how to deploy LayoutLM Document QA on Baseten.

| Property | Value |
|----------|-------|
| Task | Tutorial |
| Engine | Custom (Truss) |
| GPU | CPU |
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
  "url": "https://templates.invoicehome.com/invoice-template-us-neat-750px.png",
  "prompt": "What is the invoice number?"
}'
```

## Configuration highlights

- System packages: `tesseract-ocr`
