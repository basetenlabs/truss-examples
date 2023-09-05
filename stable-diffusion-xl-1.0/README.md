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
