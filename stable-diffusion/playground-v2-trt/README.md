# Playground v2 TensorRT Truss

Playgroun is a diffusion-based text-to-image generative model. This README covers deploying and invoking this model.

This model is packaged using [Truss](https://trussml.com), the simplest way to serve AI/ML models in production.

## Deploy Playground v2 TensorRT

First, clone this repository:

```
git clone https://github.com/basetenlabs/truss-examples/
cd stable-diffusion/playground-v2-trt
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `playground-v2-trt` as your working directory, you can deploy the model with:

```
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

Once your Truss is deployed, you can start using SDXL through the Baseten platform! Navigate to the Baseten UI to watch the model build and deploy and invoke it via the REST API.

### Hardware notes

Running inference on an A100 cuts invocation time to ~3.5 seconds.

## Invoking Playground v2 TensorRT

Playground v2 TensorRT returns an image in Base 64, which is not super useful as a string in your terminal. So we included a helpful utility script to show and save the image. Pipe the model results into the script.

```sh
truss predict -d '{"prompt": "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"}' | python show.py
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
curl -X POST "https://app.baseten.co/models/{MODEL_ID}/predict" \
     -H "Content-Type: application/json" \
     -H 'Authorization: Api-Key {YOUR_API_KEY}' \
     -d '{
           "prompt": "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
         }'
```

Again, the model will return a dictionary containing the base64-encoded image, which will need to be decoded and saved.
