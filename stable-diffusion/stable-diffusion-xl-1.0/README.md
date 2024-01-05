# Stable Diffusion XL Truss

Stable Diffusion XL 1.0 is the largest, most capable open-source image generation model of its kind. This README covers deploying and invoking this model.

This model is packaged using [Truss](https://trussml.com), the simplest way to serve AI/ML models in production.

## Deploy Stable Diffusion XL

First, clone this repository:

```
git clone https://github.com/basetenlabs/truss-examples/
cd stable-diffusion-xl-1.0
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `stable-diffusion-xl-1.0` as your working directory, you can deploy the model with:

```
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

Once your Truss is deployed, you can start using SDXL through the Baseten platform! Navigate to the Baseten UI to watch the model build and deploy and invoke it via the REST API.

### Hardware notes

Model inference runs well on an A10 with 24 GB of VRAM, with invocation time averaging ~8 seconds. If speed is essential, running inference on an A100 cuts invocation time to ~4 seconds.

## Invoking Stable Diffusion XL

Stable Diffusion XL returns an image in Base 64, which is not super useful as a string in your terminal. So we included a helpful utility script to show and save the image. Pipe the model results into the script.

```sh
truss predict -d '{"prompt": "A tree in a field under the night sky"}' | python show.py
```

The output will be a dictionary with a key `data` mapping to a base64 encoded image. It's processed with this script:

```python
import json
import base64
import os, sys

resp = sys.stdin.read()
image = json.loads(resp)["data"]
img=base64.b64decode(image)

file_name = f'{image[-10:].replace("/", "")}.jpeg'
img_file = open(file_name, 'wb')
img_file.write(img)
img_file.close()
os.system(f'open {file_name}')
```

You can also invoke your model via a REST API:

```
curl -X POST "https://app.baseten.co/models/MODEL_ID/predict" \
     -H "Content-Type: application/json" \
     -H 'Authorization: Api-Key {YOUR_API_KEY}' \
     -d '{
           "prompt": "A tree in a field under the night sky",
           "use_refiner": True
         }'
```

Again, the model will return a dictionary containing the base64-encoded image, which will need to be decoded and saved.

Here is a complete example of invoking this model in Python:

```python
import requests
import os
import base64
from PIL import Image
from io import BytesIO

# Replace the empty string with your model id below
model_id = ""
baseten_api_key = os.environ["BASETEN_API_KEY"]
BASE64_PREAMBLE = "data:image/png;base64,"

# Function used to convert a base64 string to a PIL image
def b64_to_pil(b64_str):
    return Image.open(BytesIO(base64.b64decode(b64_str.replace(BASE64_PREAMBLE, ""))))

data = {
  "prompt": "a little boy looking through a large magical portal, the boy sees a futuristic human civilization in that portal, extremely detailed, trending on artstation, 8k"
}

# Call model endpoint
res = requests.post(
    f"https://model-{model_id}.api.baseten.co/production/predict",
    headers={"Authorization": f"Api-Key {baseten_api_key}"},
    json=data
)

# Get output image
res = res.json()
output = res.get("data")

# Convert the base64 model output to an image
img = b64_to_pil(output)
img.save("output_image.png")
os.system("open output_image.png")
```

Here is the output image for the prompt shown in the request above:
![a_little_boy_looking_through_a_large_magical_portal,_the_boy_sees_a_futuristic_human_civilization_in](https://github.com/htrivedi99/truss-examples/assets/15642666/c534c752-29cb-4da8-b24e-fda6bef5876c)

Here's another example using more SDXL configurations:

```python
import requests
import os
import base64
from PIL import Image
from io import BytesIO

# Replace the empty string with your model id below
model_id = ""
baseten_api_key = os.environ["BASETEN_API_KEY"]
BASE64_PREAMBLE = "data:image/png;base64,"

# Function used to convert a base64 string to a PIL image
def b64_to_pil(b64_str):
    return Image.open(BytesIO(base64.b64decode(b64_str.replace(BASE64_PREAMBLE, ""))))

data = {
  "prompt": "Extremely detailed and intricate scene of baby phoenix hatchling cuddled up resting on a pile of ashes surrounded by fire and smoke, rays of sunlight shine on the phoenix, in the background is a dense dark forest, settings: f/8 aperture, full shot, hyper realistic, 4k",
  "negative_prompt": "worst quality, low quality",
  "width": 1248,
  "height": 832,
  "num_inference_steps": 35,
  "use_refiner": False,
  "scheduler": "DPM++ 2M",
  "guidance_scale": 14
}

# Call model endpoint
res = requests.post(
    f"https://model-{model_id}.api.baseten.co/production/predict",
    headers={"Authorization": f"Api-Key {baseten_api_key}"},
    json=data
)

# Get output image
res = res.json()
output = res.get("data")

# Convert the base64 model output to an image
img = b64_to_pil(output)
img.save("output_image.png")
os.system("open output_image.png")
```

This is the output image for the prompt above:
![Extremely_detailed_and_intricate_scene_of_baby_phoenix_hatchling_cuddled_up_resting_on_a_pile_of_ash](https://github.com/htrivedi99/truss-examples/assets/15642666/1fbf004a-9741-4c83-90a8-9e7d51dfd4e2)
