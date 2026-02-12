# FLUX.1

Deploy [FLUX.1](https://huggingface.co/black-forest-labs/FLUX.1-schnell) image generation models on Baseten. This directory contains two variants:

| Variant | Path | GPU | HuggingFace |
|---------|------|-----|-------------|
| FLUX.1 Dev | [`dev/`](dev/) | H100 40GB | [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) |
| FLUX.1 Schnell | [`schnell/`](schnell/) | H100 40GB | [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) |

## Deploy

> **Note:** This model requires a HuggingFace access token. Set `hf_access_token` in your Baseten secrets before deploying.

```sh
truss push image/flux/dev
# or
truss push image/flux/schnell
```

## Invoke

```sh
curl -X POST https://model-<model_id>.api.baseten.co/predict \
  -H "Authorization: Api-Key YOUR_BASETEN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "black forest gateau cake spelling out the words FLUX DEV, tasty, food photography, dynamic shot"}'
```

The model returns a dictionary with a `data` key containing a base64-encoded image:

```python
import requests
import base64
from PIL import Image
from io import BytesIO

res = requests.post(
    "https://model-<model_id>.api.baseten.co/predict",
    headers={"Authorization": "Api-Key YOUR_BASETEN_API_KEY"},
    json={"prompt": "A tree in a field under the night sky"},
)

output = res.json()["data"]
img = Image.open(BytesIO(base64.b64decode(output)))
img.save("output_image.jpg")
```
