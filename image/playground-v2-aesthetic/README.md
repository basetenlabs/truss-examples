# Playground V2 Aesthetic

Generate original images from text prompts.

| Property | Value |
|----------|-------|
| Model | [playgroundai/playground-v2-1024px-aesthetic](https://huggingface.co/playgroundai/playground-v2-1024px-aesthetic) |
| Task | Image generation |
| Engine | Custom (Truss) |
| GPU | A10G |
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
  "num_inference_steps": 50,
  "prompt": "A scenic mountain landscape"
}'
```

## Configuration highlights

- Engine: **Custom (Truss)**
