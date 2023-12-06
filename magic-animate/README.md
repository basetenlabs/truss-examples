# Magic Animate Truss

This repository packages [Magic Animate](https://github.com/magic-research/magic-animate) as a [Truss](https://truss.baseten.co/).

Magic Animate allows you to create human image animation using a diffusion model. The model combines two inputs, a picture of a person and a densepose motion sequence, so that the human in the picture gets animated based on the provided motion sequence. Here is an example:


https://github.com/htrivedi99/truss-examples/assets/15642666/914d7b50-c0e3-40fc-a146-c53743c82cbd


## Deploying Magic Animate

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd magic-animate
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `magic-animate` as your working directory, you can deploy the model with:

```sh
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

## Invoking the model

Here are the following inputs for the model:
1. `reference_image` (required): The image of the person you'd like to animate as a base64 string. Square images work better since each image gets resized to 512 x 512.
2. `motion_sequence` (required): A densepose motion sequence as a base64 string. You can use something like [detectron2](https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose) to create custom densepose sequences.
3. `seed` (optional): A random seed for the model.
4. `steps` (optional): The number of iterations the model runs through.
5. `guidance_scale` (optional): Used to determine how closely the image generation follows the prompt.

Here is an example of how to invoke this model:

```python
from PIL import Image
import base64

def pil_to_b64(pil_img):
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def mp4_to_base64(file_path: str):
    with open(file_path, "rb") as mp4_file:
        binary_data = mp4_file.read()
        base64_data = base64.b64encode(binary_data)
        base64_string = base64_data.decode("utf-8")

    return base64_string

def base64_to_mp4(base64_string, output_file_path):
    binary_data = base64.b64decode(base64_string)
    with open(output_file_path, "wb") as output_file:
        output_file.write(binary_data)

img = Image.open("/path/to/image/monalisa.png")
input_img = pil_to_b64(img)
motion_sequence = mp4_to_base64("/path/to/densepose/sequence/demo4.mp4")
data = {"reference_image": input_img, "motion_sequence": motion_sequence, "steps": 10}
res = requests.post("https://model-<model-id>.api.baseten.co/development/predict", headers=headers, json=data)
res = res.json()
base64_to_mp4(res.get("output"), "magic-animate.mp4")
```

Here is the example `monalisa.png` image:

![monalisa](https://github.com/htrivedi99/truss-examples/assets/15642666/9e9f4e40-6c55-415b-b37c-3271572ffb77)


Here is the example densepose sequence `demo4.mp4`:

https://github.com/htrivedi99/truss-examples/assets/15642666/c20a9761-1279-4c0b-9de1-7fd73cb43fd7
