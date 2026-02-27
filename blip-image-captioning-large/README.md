# Blip Image Captioning Large Truss

This repository packages [Blip Image Captioning Large](https://huggingface.co/Salesforce/blip-image-captioning-large) as a [Truss](https://truss.baseten.co).

Blip Image Captioning Large is a Image to Text model, specifically for generating image captions.

## Deploying Blip Image Captioning Large

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd  blip-image-captioning-large
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `blip-image-captioning-large` as your working directory, you can deploy the model with:

```sh
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

### Hardware notes

We found this model runs reasonably fast on A10Gs; you can configure the hardware you'd like in the config.yaml.

```yaml
...
resources:
  cpu: "3"
  memory: 14Gi
  use_gpu: true
  accelerator: A10G
...
```

Before deployment:

1. Make sure you have a Baseten account and API key. You can sign up for a Baseten account [here](https://app.baseten.co/signup).
2. Install Truss and the Baseten Python client: `pip install --upgrade baseten truss`
3. Authenticate your development environment with `baseten login`

Deploying the Truss is easy; simply load it and push from a Python script:

```python
import baseten
import truss

blip_image_captioning_large_truss = truss.load('.')
baseten.deploy(blip_image_captioning_large_truss)
```

## Invoking Blip Image Captioning Large

You can either choose Conditional, or Unconditional Generation

With Conditional Generation, you can set a prefix for the generated output, allowing guidance of the model:

```sh
truss predict -d '{"image_url": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg", "text" : "a photography of"}'
```

Or with the REST API

curl -X POST https://model-<Your_Model_ID>.api.baseten.co/development/predict \
  -H 'Authorization: Api-Key YOUR_API_KEY' \
  -d '{"image_url": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg", "text": "a photography of"}'



With Unconditional Generation, you let the model generate the full output on its own:

```sh
truss predict -d '{"image_url": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"}'
```

Or with the REST API

curl -X POST https://model-<Your_Model_ID>.api.baseten.co/development/predict \
  -H 'Authorization: Api-Key YOUR_API_KEY' \
  -d '{"image_url": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"}'

