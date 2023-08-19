# Stable Diffusion XL + ControlNet Truss

This Truss uses Stable Diffusion XL and ControlNet to generate images guided by input image edges. The inputs are a prompt and an image. A Canny filter is applied to the image to generate a outline, which is then passed to SDXL with the prompt.

The model is optimized for generating creative, high-fidelity images matching the input.

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
truss = truss.load(".")
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

Example request:

```json
{
  "prompt": "A painting of a cat",
  "image": "data:image/png;base64,iVBORw0KGgo...rest of base64..."
}
```

Example response:

```json
{
  "result": "data:image/png;base64,..." // generated image
}
```

Try it out with your own images or logos!
