# Stable Diffusion XL + ControlNet Depth Truss

This Truss uses Stable Diffusion XL and ControlNet with the Depth preprocessor to generate images guided by the depth map of an input image. The inputs are a prompt and an image.

## Deploying the Truss

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd sdxl-controlnet-depth-truss
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `sdxl-controlnet-depth-truss` as your working directory, you can deploy the model with:

```sh
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

## Using the model

The model takes a JSON payload with two fields:

- `prompt`: Text describing the desired image.
- `image`: Base64 encoded input image.

It returns a JSON object with the `result` field containing the generated image.

## Example Usage

You can also invoke the SDXL + ControlNet model from Python using the `baseten` SDK:

```python
import baseten
from PIL import Image
from io import BytesIO
import base64

def pil_to_b64(pil_img):
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

model = baseten.deployed_model_version_id("MODEL_VERSION_ID") # you can get this from the Baseten web UI

image = Image.open("cat.png")
image_b64 = pil_to_b64()

request = {
  "prompt": "A painting of a cat",
  "image": image_b64
}

response = model.predict(request)
```

The response will contain a base64 encoded image that you can save:

```python
def b64_to_pil(b64_str):
    return Image.open(BytesIO(base64.b64decode(b64_str.replace(BASE64_PREAMBLE, ""))))


img = b64_to_pil(response["result"])

img.save('generated.png')
```

You can also invoke the model via REST API:

```bash
curl -X POST "https://app.baseten.co/model_versions/VERSION_ID/predict" \
     -H "Content-Type: application/json" \
     -H "Authorization: Api-Key {API_KEY}" \
     -d '{"prompt": "A painting of a cat",
           "image": "data:image/png;base64,..."}'
```

The API will return a JSON response containing the generated image encoded in base64.
