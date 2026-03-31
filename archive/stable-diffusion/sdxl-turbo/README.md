# SDXL Turbo Truss

SDXL Turbo is a significant improvement over SDXL allowing for image generation in a single step while maintaining image quality. This model generates images with a latency of less than 1 second!

This model is packaged using [Truss](https://trussml.com), the simplest way to serve AI/ML models in production.

![turbo](https://github.com/htrivedi99/truss-examples/assets/15642666/904aac3e-f06d-4161-b2c6-cf48522779ce)


## Deploy SDXL Turbo

First, clone this repository:

```
git clone https://github.com/basetenlabs/truss-examples/
cd sdxl-turbo
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `sdxl-turbo` as your working directory, you can deploy the model with:

```
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

### Hardware notes

Model inference runs well on an T4 with 16 GB of VRAM, with invocation time averaging ~1 second.

## Invoking SDXL Turbo

This model only takes two inputs called `prompt` and `num_steps` and outputs a single image with the dimensions(512x512) encoded as a base 64 string.

- `prompt` (required): Text describing the desired image
- `num_steps` (optional): Number of steps the model should iterate. Must be between 1-4.

It returns a JSON object with the `result` field containing the generated image as a base 64 string.

Here is an example of how you can invoke the model using Python:

```python
BASE64_PREAMBLE = "data:image/png;base64,"
def b64_to_pil(b64_str):
    return Image.open(BytesIO(base64.b64decode(b64_str.replace(BASE64_PREAMBLE, ""))))


headers = {"Authorization": f"Api-Key <BASETEN-API-KEY>"}
resp = requests.post(
    "https://model-<model-id>.api.baseten.co/development/predict",
    headers=headers,
    json={"prompt": "An illustration of a rocket taking off an alien planet, vector art"},
)
resp = resp.json()
img = b64_to_pil(resp.get("result"))
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
