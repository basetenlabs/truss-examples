# Kaiko Midnight

Pathology foundation model for medical image analysis and classification

| Property | Value |
|----------|-------|
| Task | Embeddings |
| Engine | Custom (Truss) |
| GPU | T4 |
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
  "image_url": "https://upload.wikimedia.org/wikipedia/commons/8/80/Breast_DCIS_histopathology_%281%29.jpg",
  "task": "classification",
  "batch_size": 1
}'
```

## Configuration highlights

- Base image: `nvcr.io/nvidia/pytorch:25.06-py3`
- Predict concurrency: **32**
