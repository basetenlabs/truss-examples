# Skywork Reward Llama 3.1 8B v0.2

Deploy [Skywork/Skywork-Reward-Llama-3.1-8B-v0.2](https://huggingface.co/Skywork/Skywork-Reward-Llama-3.1-8B-v0.2) for classification using a BEI (TensorRT) engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [Skywork/Skywork-Reward-Llama-3.1-8B-v0.2](https://huggingface.co/Skywork/Skywork-Reward-Llama-3.1-8B-v0.2) |
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
