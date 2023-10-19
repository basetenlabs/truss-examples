# Stable Diffusion XL Truss with LoRA

This is a [Truss](https://truss.baseten.co/) for Stable Diffusion XL using the `DiffusionPipeline` from the `diffusers` library. This README will walk you through how to deploy this Truss on Baseten to get your own instance of Stable Diffusion XL with support for Low Rank Adaptation (LoRA).

## Overview

Stable Diffusion XL is an enhanced version of Stable Diffusion that generates 1024x1024 images. This Truss utilizes the `stabilityai/stable-diffusion-xl-base-1.0` model.

This Truss also includes support for Low Rank Adaptation (LoRA). LoRA allows you to finetune Stable Diffusion on specific prompts to improve image quality and coherence. This Truss uses the `minimaxir/sdxl-wrong-lora` weights by default, which improves image quality by reducing malformed limbs and objects.

## Deploying the Truss

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd sdxl-lora-truss
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `sdxl-lora-truss` as your working directory, you can deploy the model with:

```sh
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

## API Documentation

The API has one endpoint, `/predict`, which generates images.

**Parameters:**

- `prompt`: Input text prompt
- `size`: Image size, default 1024
- `use_refiner`: Enable/disable secondary refiner model, default true
- `high_noise_frac`: Noise level for refiner model
- `num_inference_steps`: Number of denoising steps, default 30

**Example Request:**

```json
{
  "prompt": "A beautiful painting of a fox in the forest",
  "use_refiner": true,
  "high_noise_frac": 0.8,
  "num_inference_steps": 20
}
```

**Returns:** Base64-encoded PNG image

```json
{
    "result": "..." // base64 encoded image
}

```