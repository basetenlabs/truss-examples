# Playground V2 Aesthetic Truss

This is a [Truss](https://truss.baseten.co/welcome) for [Playground V2 Aesthetic](https://huggingface.co/playgroundai/playground-v2-1024px-aesthetic)

Playground v2 is a diffusion-based text-to-image generative model. The model was trained from scratch by the research team at [Playground](https://playground.com). The research team curated 3,000 samples from Midjourney per category to train their model. According to their human preference study, people prefers the output of this model 2.5X more than Stable Diffusion XL.

Here are some examples of the outputs from this model:

![aesthetic_images](https://github.com/htrivedi99/truss-examples/assets/15642666/6c3fc815-73ec-4ab1-abab-7723884791f1)


## Deployment

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd playground-v2-aesthetic
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `playground-v2-aesthetic` as your working directory, you can deploy the model with:

```sh
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

### API route: `predict`

The predict route is the primary method for generating images based on a given prompt. It takes several parameters:

- `prompt` (required): The input text required for image generation.
- `negative_prompt` (optional): Use this to refine the image generation by discarding unwanted items.
- `scheduler` (optional): This controls which scheduler to use resulting in more image variations.
- `steps` (optional): The number of iterations the model runs through.
- `guidance_scale` (optional): Used to control how closely the image generation follows the prompt.
- `seed` (optional): Random number used to control image variations.

The output of the model is an image in the form of a base64 string.
Example model output: `{"output": "BASE64-STRING"}`

## Example usage

```sh
truss predict -d '{"prompt": "An astronaut snowboarding on an alient planet, highly detailed, 8K"}'
```

You can also invoke your model via a REST API:

```
curl -X POST " https://model-<model-id>.api.baseten.co/development/predict" \
     -H "Content-Type: application/json" \
     -H 'Authorization: Api-Key {YOUR_API_KEY}' \
     -d '{
           "prompt": "An astronaut snowboarding on an alient planet, highly detailed, 8K"
         }'
```

You can also use Python to invoke the model:

``` python
import os
import json
import base64
import requests

# Set essential values
model_id = ""
baseten_api_key = ""

# Call model endpoint
res = requests.post(
    f"https://model-{model_id}.api.baseten.co/development/predict",
    headers={"Authorization": f"Api-Key {baseten_api_key}"},
    json={
        "prompt": "a futuristic motorcycle, neon colors, cyberpunk city, detailed, 8K",
        "steps": 50
    }
)
# Get output image
res = res.json()
output = json.loads(res)["output"]
image = base64.b64decode(output)
# Save image to file
img_file = open("playground.png", "wb")
img_file.write(image)
img_file.close()
os.system("open playground.png")
```
