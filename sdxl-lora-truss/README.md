# Stable Diffusion XL Truss with LoRA

This is a [Truss](https://truss.baseten.co/) for Stable Diffusion XL using the `DiffusionPipeline` from the `diffusers` library. This README will walk you through how to deploy this Truss on Baseten to get your own instance of Stable Diffusion XL with support for Low Rank Adaptation (LoRA).

## Overview

Stable Diffusion XL is an enhanced version of Stable Diffusion that generates 1024x1024 images. This Truss utilizes the `stabilityai/stable-diffusion-xl-base-1.0` model.

This Truss also includes support for Low Rank Adaptation (LoRA). LoRA allows you to finetune Stable Diffusion on specific prompts to improve image quality and coherence. This Truss uses the `minimaxir/sdxl-wrong-lora` weights by default, which improves image quality by reducing malformed limbs and objects.

## Deploying the Truss

To deploy this Truss, follow these steps:

1. Sign up for a Baseten account and get your API key from the dashboard.

2. Install Truss and the Baseten Python client:

```
pip install baseten truss
```

3. Load the Truss into an IPython shell:

```python
import truss

truss = truss.load("path/to/truss")
```

4. Log in to Baseten with your API key:

```python
import baseten

baseten.login("YOUR_API_KEY")
```

5. Deploy the Truss:

```python
baseten.deploy(truss)
```

Once deployed, you can access Stable Diffusion XL via the Baseten UI and API.

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