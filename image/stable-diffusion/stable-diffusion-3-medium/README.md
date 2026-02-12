# Stable Diffusion 3 Medium

Deploy [stabilityai/stable-diffusion-3-medium-diffusers](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers) for image generation on Baseten.

| Property | Value |
|----------|-------|
| Model | [stabilityai/stable-diffusion-3-medium-diffusers](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers) |
| Task | Image generation |
| Engine | Custom (Truss) |
| GPU | A100 |
| Python | py310 |

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

- System packages: `ffmpeg, libsm6, libxext6`
- Environment variables: `HF_HUB_OFFLINE`
