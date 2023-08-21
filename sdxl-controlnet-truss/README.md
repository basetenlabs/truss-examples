# Stable Diffusion XL + ControlNet Truss

This Truss uses Stable Diffusion XL and ControlNet to generate images guided by input image edges. The inputs are a prompt and an image. A Canny filter is applied to the image to generate a outline, which is then passed to SDXL with the prompt.

![baseten_controlnet](baseten-logo.gif)

## Deploying the Truss

To deploy this Truss:

1. Sign up for a Baseten account if you don't already have one.

2. Install the Baseten Python client:

```
pip install baseten
```

3. Load the Truss into memory:

```python
import truss
truss = truss.load("/path/to/sdxl_controlnet_truss")
```

4. Log in to Baseten and deploy:

```python
import baseten

baseten.login("YOUR_API_KEY")
baseten.deploy(truss)
```

Once deployed, you can access the model via the Baseten UI or API.

## Using the model

The model takes a JSON payload with two fields:

- `prompt`: Text describing the desired image.
- `image`: Base64 encoded input image.

It returns a JSON object with the `result` field containing the generated image.

## Example Usage

You can invoke the SDXL + ControlNet model from Python using the `baseten` SDK:

```python
import baseten

model = baseten.deployed_model_version("MODEL_VERSION_ID") # you can get this from the Baseten web UI

image = open("cat.png", "rb").read()
image_b64 = base64.b64encode(image).decode("utf-8")

request = {
  "prompt": "A painting of a cat",
  "image": "data:image/png;base64," + image_b64
}

response = model.predict(request)
```

The response will contain a base64 encoded image that you can save:

```python
import base64

img = base64.b64decode(response["data"])

with open("generated.png", "wb") as f:
  f.write(img)
```

You can also invoke the model via REST API:

```bash
curl -X POST "https://app.baseten.co/model_versions/VERSION_ID/predict" \
     -H "Content-Type: application/json" \
     -H "Authorization: Api-Key {API_KEY}" \
     -d '{"prompt": "A painting of a cat",
           "image": "data:image/png;base64,..."}'
```

The API will return a JSON response containing the generated image encoded in base64.
