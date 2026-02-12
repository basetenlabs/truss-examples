# Flux 1.0 Dev - TensorRT

Generate high-quality images from text prompts using Black Forest Labs's Flux model with TensorRT optimization.

| Property | Value |
|----------|-------|
| Task | Image generation |
| Engine | Custom (Truss) |
| GPU | B200 |

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
  -d '{"prompt": "A photo of a cat in a field of sunflowers"}'
```

> The response may contain base64-encoded image data.

## Configuration highlights

- Base image: `nvcr.io/nvidia/pytorch:25.06-py3`
- Predict concurrency: **1**
- System packages: `ffmpeg, libsm6, libxext6`
