# Stable Diffusion Truss

This is a [Truss](https://truss.baseten.co/) for Stable Diffusion v2.1 using the `StableDiffusionPipeline` from the `diffusers` library. This README will walk you through how to deploy this Truss on Baseten to get your own instance of the Stable Diffusion.

## Deploy Stable Diffusion

First, clone this repository:

```
git clone https://github.com/basetenlabs/truss-examples/
cd stable-diffusion-truss
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `stable-diffusion-truss` as your working directory, you can deploy the model with:

```
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

Once your Truss is deployed, you can start using Stable Diffusion through the Baseten platform! Navigate to the Baseten UI to watch the model build and deploy and invoke it via the REST API.

## Invoking Stable Diffusion

Stable Diffusion returns an image in Base 64, which is not super useful as a string in your terminal. So we included a helpful utility script to show and save the image. Pipe the model results into the script.

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
           "prompt": "A tree in a field under the night sky"
         }'
```

Again, the model will return a dictionary containing the base64-encoded image, which will need to be decoded and saved.

### Stable Diffusion API documentation

This section provides an overview of the Stable Diffusion API, its parameters, and how to use it. The API consists of a single route named `predict`, which you can invoke to generate images based on the provided parameters.

#### API route: `predict`

The predict route is the primary method for generating images based on a given set of parameters. It takes several parameters:

- **prompt**: The input text you'd like to generate an image for
- **scheduler**: (optional, default: DDIM) The scheduler used for the diffusion process. Choose from: "ddim", "dpm", "euler", "lms", or "pndm".
- **seed**: (optional) A random seed for deterministic results. If not provided, a random seed will be generated.
- **negative_prompt**: (optional) A string representing the negative prompt, or prompts that indicate what you don't want to generate.

The API also supports passing any parameter supported by Diffuser's `StableDiffusionPipeline`.
