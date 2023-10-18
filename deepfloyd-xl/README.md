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
3. Set your Huggingface token as a Baseten secret [here](https://app.baseten.co/settings/secrets) with the key `hf_access_token`.

## Deploying DeepFloyd XL

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd deepfloyd-xl-truss
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `deepfloyd-xl-truss` as your working directory, you can deploy the model with:

```sh
truss push --trusted
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

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

```sh
truss predict -d '{"prompt": "man on moon"}'
```

You can also invoke it via cURL:

```sh
curl -X POST https://app.baseten.co/models/EqwKvqa/predict \
  -H 'Authorization: Api-Key {YOUR_API_KEY}' \
  -d '{"prompt": "man on moon"}'
```
