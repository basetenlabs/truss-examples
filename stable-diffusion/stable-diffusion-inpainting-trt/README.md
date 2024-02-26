# Stable Diffusion Inpainting TensorRT

This is a [Truss](https://truss.baseten.co/) for Stable Diffusion v1.5 Inpainting. It has been optimized to increase performance using TensorRT.

## Deploy Stable Diffusion Inpainting TensorRT

First, clone this repository:

```
git clone https://github.com/basetenlabs/truss-examples/
cd stable-diffusion/stable-diffusion-inpainting-trt
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `stable-diffusion-inpainting-trt` as your working directory, you can deploy the model with:

```
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

Once your Truss is deployed, you can start using Stable Diffusion through the Baseten platform! Navigate to the Baseten UI to watch the model build and deploy and invoke it via the REST API.

## Invoking Stable Diffusion Inpainting TensorRT

The model accepts a few inputs:
- __prompt__(required): Text describing the output image.
- __negative_prompt__(optional): Text used to steer the model away from undesired output.
- __image__(required): The input image used for inpainting in the form of a base64 string. The image should be 512 x 512 px in size.
- __mask__(required): The an image representing the mask or the area that you want stable diffusion to generate over. It should also be a base64 string and the same dimensions as the input image.

```python
from PIL import Image
import base64
import requests

BASE64_PREAMBLE = "data:image/png;base64,"

def pil_to_b64(pil_img):
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def b64_to_pil(b64_str):
    return Image.open(BytesIO(base64.b64decode(b64_str.replace(BASE64_PREAMBLE, ""))))


data = {
    "prompt": "A tiger",
    "image": pil_to_b64(Image.open("/path/to/image/dog.png")),
    "mask": pil_to_b64(Image.open("/path/to/mask/mask.png"))
}

headers = {"Authorization": "Api-Key <BASETEN-API-KEY>"}

res = requests.post(
    "https://model-<model-id>.api.baseten.co/development/predict",
    headers=headers,
    json=data,
)

res = res.json()
outputs = res.get("outputs")

for out in outputs:
    img = b64_to_pil(out)
    img.show()
```
