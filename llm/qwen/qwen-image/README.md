# Qwen Image

Deploy [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) for text generation using a Custom (Truss) engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) |
| Task | Text generation |
| Engine | Custom (Truss) |
| GPU | H100 |
| Python | py311 |

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
  "prompt": "A beautiful sunset over a mountain landscape with golden clouds, Ultra HD, 4K, cinematic composition",
  "width": 1024,
  "height": 1024,
  "num_inference_steps": 50,
  "true_cfg_scale": 4.0,
  "seed": 42
}'
```

## Configuration highlights

- System packages: `ffmpeg, libsm6, libxext6`
