# SD Turbo

SD Turbo is a fast text-to-image model built for real time image synthesis. It is a distilled version of Stable Diffusion 2.1.

SD Turbo is a smaller model than [SDXL Turbo](https://github.com/basetenlabs/truss-examples/tree/main/stable-diffusion/sdxl-turbo). Comparatively, it achieves faster latency and tends to produce outputs of lower quality and prompt alignment.

## Deploying SD Turbo

First, clone this repository and navigate to the `sd-turbo` directory:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd sd-turbo
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `sd-turbo` as your working directory, deploy the model with:

```sh
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

### Hardware notes

On a T4 with 16 GiB VRAM, we have observed that the SD Turbo mean inference time is < 0.5 seconds.

## Invoking SD Turbo

The SD Turbo truss takes in a JSON payload with two fields:
- `prompt` (required): Text describing the desired image.
- `num_steps` (optional, default = 1): Number of steps the model should iterate. Must be between 1-4.

It returns a JSON object with a `result` field, which contains a generated image of size 512 x 512 encoded as a base 64 string.

### Example usage

The example code below invokes the SD Turbo truss, parses the image from the response, and saves the image.

```python
from PIL import Image
from io import BytesIO
import base64
import requests

BASE64_PREAMBLE = "data:image/png;base64,"

def b64_to_pil(b64_str):
    return Image.open(BytesIO(base64.b64decode(b64_str.replace(BASE64_PREAMBLE, ""))))

# Paste your model ID here. This can be grabbed from the Baseten UI or from the
# output of `truss push`.
MODEL_ID = ""
# Development model endpoint URL. To call the production deployment or another
# deployment, replace this with the desired endpoint URL from the Baseten UI.
MODEL_ENDPOINT = f"https://model-{MODEL_ID}.api.baseten.co/development/predict"

# Paste your Baseten API key here.
API_KEY = ""
HEADERS = {"Authorization": f"Api-Key {API_KEY}"}

resp = requests.post(
    MODEL_ENDPOINT,
    headers=HEADERS,
    json={"prompt": "A tree in a field under the night sky"},
)
resp = resp.json()
img = b64_to_pil(resp.get("result"))

# Save the image.
img.save("sd_turbo_output.png")
```