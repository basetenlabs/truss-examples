[![Deploy to Baseten](https://user-images.githubusercontent.com/2389286/236301770-16f46d4f-4e23-4db5-9462-f578ec31e751.svg)](https://app.baseten.co/explore/stable_diffusion)

# Stable Diffusion Truss

This is a [Truss](https://truss.baseten.co/) for Stable Diffusion v2.1 using the `StableDiffusionPipeline` from the `diffusers` library. This README will walk you through how to deploy this Truss on Baseten to get your own instance of the Stable Diffusion.

## Truss

Truss is an open-source model serving framework developed by Baseten. It allows you to develop and deploy machine learning models onto Baseten (and other platforms like [AWS](https://truss.baseten.co/deploy/aws) or [GCP](https://truss.baseten.co/deploy/gcp). Using Truss, you can develop a GPU model using [live-reload](https://baseten.co/blog/technical-deep-dive-truss-live-reload), package models and their associated code, create Docker containers and deploy on Baseten.

## Deploying Stable Diffusion

To deploy the Stable Diffusion Truss, you'll need to follow these steps:

1. __Prerequisites__: Make sure you have a Baseten account and API key. You can sign up for a Baseten account [here](https://app.baseten.co/signup).

2. __Install Truss and the Baseten Python client__: If you haven't already, install the Baseten Python client and Truss in your development environment using:
```
pip install --upgrade baseten truss
```

3. __Load the Stable Diffusion Truss__: Assuming you've cloned this repo, spin up an IPython shell and load the Truss into memory:
```
import truss
stable_diffusion_truss = truss.load("path/to/stable_diffusion_truss")
```

4. __Log in to Baseten__: Log in to your Baseten account using your API key (key found [here](https://app.baseten.co/settings/account/api_keys)):
```
import baseten

baseten.login("PASTE_API_KEY_HERE")
```

5. __Deploy the Stable Diffusion Truss__: Deploy Stable Diffusion to Baseten with the following command:
```
baseten.deploy(stable_diffusion_truss)
```

Once your Truss is deployed, you can start using Stable Diffusion through the Baseten platform! Navigate to the Baseten UI to watch the model build and deploy and invoke it via the REST API.

## Stable Diffusion API documentation
This section provides an overview of the Stable Diffusion API, its parameters, and how to use it. The API consists of a single route named `predict`, which you can invoke to generate images based on the provided parameters.

### API route: `predict`
The predict route is the primary method for generating images based on a given set of parameters. It takes several parameters:

- __prompt__: The input text you'd like to generate an image for
- __scheduler__: (optional, default: DDIM) The scheduler used for the diffusion process. Choose from: "ddim", "dpm", "euler", "lms", or "pndm".
- __seed__: (optional) A random seed for deterministic results. If not provided, a random seed will be generated.
- __negative_prompt__: (optional) A string representing the negative prompt, or prompts that indicate what you don't want to generate.

The API also supports passing any parameter supported by Diffuser's `StableDiffusionPipeline`.

## Example usage

You can use the `baseten` model package to invoke your model from Python

```python
import baseten

# You can retrieve your deployed model version ID from the UI
model = baseten.deployed_model_version_id('MODEL_VERSION_ID')

request = {
    "prompt": "man on moon",
    "scheduler": "ddim",
    "negative_prompt": "disfigured hands"
}

response = model.predict(request)
```

The output will be a dictionary with a key `data` mapping to a list with a base64 encoding of the image. You can save the image with the following snippet:

```python
import base64

img=base64.b64decode(response["data"][0])

img_file = open('image.jpeg', 'wb')
img_file.write(img)
img_file.close()
```

You can also invoke your model via a REST API
```
curl -X POST "https://app.baseten.co/models/YOUR_MODEL_ID/predict" \
     -H "Content-Type: application/json" \
     -H 'Authorization: Api-Key {YOUR_API_KEY}' \
     -d '{
           "prompt" : "man on moon",
           "scheduler": "ddim",
           "negative_prompt" : "disfigured hands"
         }'
```

Again, the model will return a dictionary containing the base64-encoded image, which will need to be decoded and saved.
