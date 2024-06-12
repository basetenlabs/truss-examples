# Stable Diffusion 3 Medium Truss

This is a truss for the brand new Stable Diffusion 3 Medium model.

## Deploy Stable Diffusion 3

First, clone this repository:

```
git clone https://github.com/basetenlabs/truss-examples/
cd stable-diffusion/stable-diffusion-3-medium
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `stable-diffusion-3-medium` as your working directory, you can deploy the model with:

```
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

Once your Truss is deployed, you can start using Stable Diffusion through the Baseten platform! Navigate to the Baseten UI to watch the model build and deploy and invoke it via the REST API.

## Invoking Stable Diffusion 3

The output will be a dictionary with a key `data` mapping to a base64 encoded image. It's processed with this script:

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

You can also invoke your model via a REST API:

```
curl -X POST "https://model-{model_id}.api.baseten.co/production/predict" \
     -H "Content-Type: application/json" \
     -H 'Authorization: Api-Key {YOUR_API_KEY}' \
     -d '{
           "prompt": "A tree in a field under the night sky"
         }'
```

Again, the model will return a dictionary containing the base64-encoded image, which will need to be decoded and saved.
