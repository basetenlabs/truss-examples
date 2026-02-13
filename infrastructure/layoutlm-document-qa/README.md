# LayoutLM Document QA

Extract information from images of invoices

| Property | Value |
|----------|-------|
| Task | Infrastructure / Custom server |
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
  "prompt": "What is the invoice number?",
  "url": "https://templates.invoicehome.com/invoice-template-us-neat-750px.png"
}'
```

## Configuration highlights

- System packages: `tesseract-ocr`
