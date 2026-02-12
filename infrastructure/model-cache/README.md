# Model Cache

Demonstrates how to use Truss `model_cache` to cache model weights across deployments for faster cold starts. This example caches Stable Diffusion XL weights using volume-mounted HuggingFace repos.

| Property | Value |
|----------|-------|
| Task | Infrastructure / Model caching |
| Engine | Custom (Truss) |
| GPU | CPU |
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
  -d '{"text": "hello"}'
```

The model returns the input data along with the size of the cached model weights file.

## Configuration highlights

- Model cache: **volume-mounted** for fast cold starts
- Two cached HuggingFace repos: `madebyollin/sdxl-vae-fp16-fix` and `stabilityai/stable-diffusion-xl-base-1.0`
- Uses `revision` pinning and `allow_patterns` for selective downloads
- `runtime_secret_name` for authenticated HF access at runtime
