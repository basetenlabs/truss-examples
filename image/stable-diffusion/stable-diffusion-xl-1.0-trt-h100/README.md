# Stable Diffusion XL - TensorRT

Generate original images from text prompts.

| Property | Value |
|----------|-------|
| Model | [baseten/sdxl-1.0-trt-8.6.1.post1-engine-H100](https://huggingface.co/baseten/sdxl-1.0-trt-8.6.1.post1-engine-H100) |
| Task | Image generation |
| Engine | Custom (Truss) |
| GPU | H100 |
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
  "prompt": "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
}'
```

## Configuration highlights

- Base image: `nvcr.io/nvidia/pytorch:23.11-py3`
- Predict concurrency: **1**
- System packages: `python3.10-venv, ffmpeg, libsm6, libxext6`
- Environment variables: `HF_HUB_ENABLE_HF_TRANSFER`
