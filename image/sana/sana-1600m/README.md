# Sana 1600M

Deploy Sana 1600M for image generation on Baseten.

| Property | Value |
|----------|-------|
| Task | Image generation |
| Engine | Custom (Truss) |
| GPU | H100_40GB |
| Python | py311 |

## Deploy

> **Note:** This model requires a HuggingFace access token. Set `hf_access_token` in your Baseten secrets before deploying.

```sh
truss push
```

## Invoke

```sh
curl -X POST https://model-<model_id>.api.baseten.co/predict \
  -H "Authorization: Api-Key YOUR_BASETEN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "prompt": "a photo of an astronaut riding a horse on mars",
  "height": 1024,
  "width": 1024,
  "guidance_scale": 5.0,
  "pag_guidance_scale": 2.0,
  "num_inference_steps": 18,
  "seed": 4096
}'
```

## Configuration highlights

- Base image: `alphatozeta/cuda-python:12.1.1-cudnn8-devel-ubuntu22.04`
- System packages: `ffmpeg, libsm6, libxext6, python3.10-venv`
