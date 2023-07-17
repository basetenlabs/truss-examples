[![Deploy to Baseten](https://user-images.githubusercontent.com/2389286/236301770-16f46d4f-4e23-4db5-9462-f578ec31e751.svg)](https://app.baseten.co/explore/deepfloyd)

# DeepFloyd XL Truss

This is a [Truss](https://truss.baseten.co/) for DeepFloyd-IF. DeepFloyd-IF is a pixel-based text-to-image triple-cascaded diffusion model that can generate pictures and sets a new state-of-the-art for photorealism and language understanding. The result is a highly efficient model that outperforms current state-of-the-art models, achieving a zero-shot FID-30K score of 6.66 on the COCO dataset.

Model details:

- Developed by: DeepFloyd, StabilityAI
- Model type: pixel-based text-to-image cascaded diffusion model
- Cascade Stage: I
- Num Parameters: 4.3B
- Language(s): primarily English and, to a lesser extent, other Romance languages
- License: [DeepFloyd IF License Agreement](https://huggingface.co/spaces/DeepFloyd/deepfloyd-if-license)
- Model Description: DeepFloyd-IF is modular composed of frozen text mode and three pixel cascaded diffusion modules, each designed to generate images of increasing resolution: 64x64, 256x256, and 1024x1024. All stages of the model utilize a frozen text encoder based on the T5 transformer to extract text embeddings, which are then fed into a UNet architecture enhanced with cross-attention and attention-pooling

Before deploying this model, you'll need to:

1. Accept the terms of service of the Deepfloyd XL model [here](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0).
2. Retrieve your Huggingface token from the [settings](https://huggingface.co/settings/tokens).
3. Set your Huggingface token as a Baseten secret [here](https://app.baseten.co/settings/secrets) with the key `hf_api_key`.

## Deploying DeepFloyd XL

To deploy the DeepFloyd XL Truss, you'll need to follow these steps:

1. __Prerequisites__: Make sure you have a Baseten account and API key. You can sign up for a Baseten account [here](https://app.baseten.co/signup).

2. __Install Truss and the Baseten Python client__: If you haven't already, install the Baseten Python client and Truss in your development environment using:
```
pip install --upgrade baseten truss
```

3. __Load the DeepFloyd XL Truss__: Assuming you've cloned this repo, spin up an IPython shell and load the Truss into memory:
```
import truss

deepfloyd_truss = truss.load("path/to/deepfloyd_truss")
```

4. __Log in to Baseten__: Log in to your Baseten account using your API key (key found [here](https://app.baseten.co/settings/account/api_keys)):
```
import baseten

baseten.login("PASTE_API_KEY_HERE")
```

5. __Deploy the DeepFloyd XL Truss__: Deploy the DeepFloyd XL Truss to Baseten with the following command:
```
baseten.deploy(deepfloyd_truss)
```

Once your Truss is deployed, you can start using the DeepFloyd XL model through the Baseten platform! Navigate to the Baseten UI to watch the model build and deploy and invoke it via the REST API.

## DeepFloyd API documentation

### Input

This deployment of DeepFloyd takes a dictionary as input, which requires the following key:

* `prompt` - the prompt for image generation

It also supports a number of other parameters detailed in [this blog post](https://huggingface.co/blog/if).

### Output

The result will be a dictionary containing:

* `status` - either `success` or `failed`
* `data` - list of base 64 encoded images
* `message` - will contain details in the case of errors

```json
{"status": "success", "data": ["/9j/4AAQSkZJRgABAQAAAQABAA...."], "message": null}
```

## Example usage

```
curl -X POST https://app.baseten.co/models/EqwKvqa/predict \
  -H 'Authorization: Api-Key {YOUR_API_KEY}' \
  -d '{"prompt": "man on moon"}'
```
