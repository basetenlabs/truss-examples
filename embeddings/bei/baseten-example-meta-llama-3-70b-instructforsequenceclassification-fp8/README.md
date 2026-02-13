# Baseten Example Meta Llama 3 70B InstructForSequenceClassification

Deploy [baseten/example-Meta-Llama-3-70B-InstructForSequenceClassification](https://huggingface.co/baseten/example-Meta-Llama-3-70B-InstructForSequenceClassification) for classification using a BEI (TensorRT) engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [baseten/example-Meta-Llama-3-70B-InstructForSequenceClassification](https://huggingface.co/baseten/example-Meta-Llama-3-70B-InstructForSequenceClassification) |
| Task | Classification |
| Engine | BEI (TensorRT) |
| GPU | H100 |
| Quantization | FP8 |
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

- Quantization: **fp8**
