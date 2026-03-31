# DIS Segmentation Truss

This is a [Truss](https://truss.baseten.co/welcome) for [DIS](https://github.com/xuebinqin/DIS/tree/main)

This model can be used to remove backgrounds from an image or create a segmentation mask for an object.

This model takes an input image and creates two output images:
1. This first output image is the original image with the background removed
2. The second output image contains the mask of the object in the foreground


## Deployment

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd dis-segmentation
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `dis-segmentation` as your working directory, you can deploy the model with:

```sh
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

### API route: `predict`

The model only has one input:

- `input_image` (required): An image represented as a base74 string.

The output of the model contains two images in the form of a base64 string.
Example model output:
```json
{
    "img_without_bg": "Base64 string image",
    "image_mask": "Base64 string image"
}
```

## Example usage

You can invoke the model in Python like so:

``` python
import os
import json
import base64
import requests

# Set essential values
model_id = ""
baseten_api_key = ""

BASE64_PREAMBLE = "data:image/png;base64,"

def pil_to_b64(pil_img):
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def b64_to_pil(b64_str):
    return Image.open(BytesIO(base64.b64decode(b64_str.replace(BASE64_PREAMBLE, ""))))

# Call model endpoint
res = requests.post(
    f"https://model-{model_id}.api.baseten.co/development/predict",
    headers={"Authorization": f"Api-Key {baseten_api_key}"},
    json={
        "input_image": pil_to_b64(Image.open("path/to/image.png"))
    }
)
# Get output image
res = res.json()
background_removed = b64_to_pil(res.get("img_without_bg"))
mask = b64_to_pil(res.get("image_mask"))
background_removed.save("image_without_background.png")
mask.save("image_mask.png")
```

Here is an example of the outputs given the following input.

Input image:

![input image](https://github.com/basetenlabs/truss-examples/assets/15642666/a0491a6f-795b-4e2a-aa66-8830f4fb86b3)

Image without background:

![removed-bg](https://github.com/basetenlabs/truss-examples/assets/15642666/a94ceba8-9a62-4e38-9fbe-f6f9382c8086)

Image mask:

![mask](https://github.com/basetenlabs/truss-examples/assets/15642666/14bcaea1-2136-4cdb-b896-bbed9e99fa89)
