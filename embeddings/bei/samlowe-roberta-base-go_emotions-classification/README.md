# SamLowe RoBERTa Base GoEmotions Classification

Deploy [SamLowe/roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions) for classification using a BEI (TensorRT) engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [SamLowe/roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions) |
| Task | Classification |
| Engine | BEI (TensorRT) |
| GPU | L4 |
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
  "inputs": [
    [
      "Baseten is a fast inference provider"
    ],
    [
      "Classify this separately."
    ]
  ],
  "raw_scores": true,
  "truncate": true,
  "truncation_direction": "Right"
}'
```

## Configuration highlights

- Engine: **BEI (TensorRT)**
