# LLaVA v1.6 SGL Truss

This is a truss to run [Llava 1.6 using SGL](https://github.com/sgl-project/sglang)

## Deploying LLaVA

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd llava/llava-v1.6-sgl
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `llava/llava-v1.6-sgl` as your working directory, you can deploy the model with:

```sh
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

## Invoking LLaVA

LLaVA takes in the following inputs:
 __prompt__(required): Piece of text used as instruction for the LLM
 __image__(required): Input image in the form of a base64 string used by the model
 __max_new_tokens__(optional): Max number of output tokens generated by the LLM
 __temperature__(optional): Configuration for LLM

LLaVA will respond to the `prompt` conditioned on the `image`. The output is is a stream of tokens containing the model response.


```python
from PIL import Image
from io import BytesIO
import base64
import requests


def pil_to_b64(pil_img):
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

data = {
    "prompt": "What is this a picture of?",
    "image": pil_to_b64(Image.open("/path/to/image/mountain.jpeg")),
}

# Call model endpoint
res = requests.post(
    f"https://model-<model-id>.api.baseten.co/production/predict",
    headers=headers,
    json=data,
    stream=True
)

# Print the generated tokens as they get streamed
for content in res.iter_content():
    print(content.decode("utf-8"), end="", flush=True)
```

Sample Input:
![mountain](https://github.com/basetenlabs/truss-examples/assets/15642666/5eb63370-0296-40ab-9387-428bf5e3cd53)

Sample output:
```
This is a picture of Half Dome, a granite dome located in Yosemite National Park in the Sierra Nevada of California. It is one of the most iconic rock formations in the park and a popular destination for hikers and climbers. The image shows the dome with a clear blue sky and some clouds, highlighting the natural beauty of the area.
```
