# SDXL Lightning Truss

SDXL Turbo is a significant improvement over SDXL allowing for image generation in a single step while maintaining image quality. This model generates images with a latency of less than 1 second!

This model is packaged using [Truss](https://trussml.com), the simplest way to serve AI/ML models in production.

## Deploy SDXL Lightning

First, clone this repository:

```
git clone https://github.com/basetenlabs/truss-examples/
cd stable-diffusion/sdxl-lightning
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `stable-diffusion/sdxl-lightning` as your working directory, you can deploy the model with:

```
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

### Hardware notes

Model inference runs well on an A100.

## Invoking SDXL Turbo

This model only takes two inputs called `prompt` and `num_steps` and outputs a single image with the dimensions(512x512) encoded as a base 64 string.

- `prompt` (required): Text describing the desired image

It returns a JSON object with the `result` field containing the generated image as a base 64 string.

Here is an example of how you can invoke the model using Python:

```python
import base64
import requests
import os

# Replace the empty string with your model id below
model_id = ""
baseten_api_key = os.environ["BASETEN_API_KEY"]
BASE64_PREAMBLE = "data:image/png;base64,"

data = {
  "prompt": "a picture of a rhino wearing a suit",
}

# Call model endpoint
res = requests.post(
    f"https://model-{model_id}.api.baseten.co/production/predict",
    headers={"Authorization": f"Api-Key {baseten_api_key}"},
    json=data
)

# Get output image
res = res.json()
img_b64 = res.get("result")
img = base64.b64decode(img_b64)

# Save the base64 string to a PNG
img_file = open("sdxl-output-1.png", "wb")
img_file.write(img)
img_file.close()
os.system("open sdxl-output-1.png")
```

You can also invoke your model via a REST API:

```
curl -X POST "https://model-<model-id>.api.baseten.co/development/predict" \
     -H "Content-Type: application/json" \
     -H 'Authorization: Api-Key {YOUR_API_KEY}' \
     -d '{
           "prompt": "An illustration of a rocket taking off an alien planet, vector art"
         }'
```
