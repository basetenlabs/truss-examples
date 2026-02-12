# AllenAI Llama 3.1 Tulu 3 8B Reward Model

Deploy [allenai/Llama-3.1-Tulu-3-8B-RM](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B-RM) for classification using a BEI (TensorRT) engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [allenai/Llama-3.1-Tulu-3-8B-RM](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B-RM) |
| Task | Classification |
| Engine | BEI (TensorRT) |
| GPU | H100_40GB |
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
