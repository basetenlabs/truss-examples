# Stable Diffusion XL Truss

Stable Diffusion XL 1.0 is the largest, most capable open-source image generation model of its kind. This README covers deploying and invoking this model.

This model is packaged using [Truss](https://trussml.com), the simplest way to serve AI/ML models in production.

## Setup

[Sign up](https://app.baseten.co/signup) or [sign in](https://app.baseten.co/login/) to your Baseten account and create an [API key](https://app.baseten.co/settings/account/api_keys).

Then run:

```
pip install --upgrade baseten
baseten login
```

Paste your API key when prompted.

## Deployment

First, clone this repository:

```
git clone https://github.com/basetenlabs/truss-examples/
```

Then, in an iPython notebook, run the following script to deploy SDXL to your Baseten account:

```python
import baseten
import truss

sdxl = truss.load("truss-examples/stable-diffusion-xl-1.0")
baseten.deploy(
  sdxl,
  model_name="Stable Diffusion XL 1.0"
)
```

Once your Truss is deployed, you can start using SDXL through the Baseten platform! Navigate to the Baseten UI to watch the model build and deploy and invoke it via the REST API.

### Hardware notes

Model inference runs well on an A10 with 24 GB of VRAM, with invocation time averaging ~16 seconds. If speed is essential, running inference on an A100 cuts invocation time to ~8 seconds.

## Example usage

You can use the `baseten` model package to invoke your model from Python

```python
import baseten

# You can retrieve your deployed model version ID from the UI
model = baseten.deployed_model_version_id('MODEL_VERSION_ID')

request = {
    "prompt": "A tree in a field under the night sky",
    "use_refiner": True
}

response = model.predict(request)
```

The output will be a dictionary with a key `data` mapping to a base64 encoded image. You can save the image with the following snippet:

```python
import base64

img=base64.b64decode(response["data"])

img_file = open('image.jpeg', 'wb')
img_file.write(img)
img_file.close()
```

You can also invoke your model via a REST API:

```
curl -X POST "https://app.baseten.co/model_versions/YOUR_MODEL_VERSION_ID/predict" \
     -H "Content-Type: application/json" \
     -H 'Authorization: Api-Key {YOUR_API_KEY}' \
     -d '{
           "prompt": "A tree in a field under the night sky",
           "use_refiner": True
         }'
```

Again, the model will return a dictionary containing the base64-encoded image, which will need to be decoded and saved.
