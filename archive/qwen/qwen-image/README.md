# Qwen Image Model

![Qwen Image on Baseten](assets/generated_image.jpg)

This Truss serves the Qwen Image model, a powerful text-to-image generation model that supports both English and Chinese prompts. The model is based on the Qwen/Qwen-Image model from Hugging Face and is optimized for high-quality image generation. It is Apache 2.0 licensed and can be used commercially without restrictions.

## Model Description

The Qwen Image model is a diffusion-based text-to-image model that can generate high-quality images from text prompts. It features:

- **Multilingual Support**: Handles both English and Chinese prompts
- **High Quality**: Generates 4K quality images with cinematic composition
- **Flexible Aspect Ratios**: Supports various image dimensions
- **Customizable Parameters**: Adjustable inference steps, guidance scale, and more

## Model Parameters

The model accepts the following parameters:

- `prompt` (required): The text description of the image you want to generate
- `negative_prompt` (optional): Text describing what you don't want in the image (default: "")
- `width` (optional): Image width in pixels (default: 1024)
- `height` (optional): Image height in pixels (default: 1024)
- `num_inference_steps` (optional): Number of denoising steps (default: 50)
- `true_cfg_scale` (optional): Guidance scale for generation (default: 4.0)
- `seed` (optional): Random seed for reproducible results (default: random)

## Example Usage

The model outputs a base64 string which can be saved locally.

```python
import httpx
import os
import base64
from PIL import Image
from io import BytesIO

# Replace with your model ID and API key
model_id = "your-model-id"
baseten_api_key = os.environ["BASETEN_API_KEY"]

def b64_to_pil(b64_str):
    """Convert base64 string to PIL image"""
    return Image.open(BytesIO(base64.b64decode(b64_str)))

# Example 1: English prompt
english_data = {
    "prompt": "A fashionably dressed man on the streets of New York City holds a sign that says `Qwen Image on Baseten`",
    "width": 1664,
    "height": 928,
    "num_inference_steps": 50,
    "seed": 42
}

# Example 2: Chinese prompt
chinese_data = {
    "prompt": "一只可爱的小猫坐在花园里，阳光明媚",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 50
}

# Call the model with extended timeout for image generation
print("Generating image... This may take a moment.")
response = httpx.post(
    f"https://model-{model_id}.api.baseten.co/development/predict",
    headers={"Authorization": f"Api-Key {baseten_api_key}"},
    json=english_data,
    timeout=httpx.Timeout(120.0)
)

# Get the result
result = response.json()
image_b64 = result.get("data")

# Convert to image and save
image = b64_to_pil(image_b64)
image.save("generated_image.jpg")
print("Image generated successfully! Saved as 'generated_image.jpg'")
```

## Aspect Ratio Examples

The model supports various aspect ratios. Here are some common configurations:

```python
# Square (1:1)
{"width": 1024, "height": 1024}

# Landscape (16:9)
{"width": 1664, "height": 928}

# Portrait (9:16)
{"width": 928, "height": 1664}

# Traditional (4:3)
{"width": 1472, "height": 1140}

# Portrait Traditional (3:4)
{"width": 1140, "height": 1472}
```

## Deployment

To deploy this model:

1. Clone the repository
2. Make sure you have the Truss CLI installed (`pip install truss`)
3. Run the deployment command:

```bash
truss push qwen/qwen-image --publish
```

## Model Features

- **Automatic Language Detection**: The model automatically detects Chinese vs English prompts and applies appropriate quality enhancements
- **Quality Enhancement**: Automatically adds "Ultra HD, 4K, cinematic composition" for English prompts or "超清，4K，电影级构图" for Chinese prompts
- **GPU Optimization**: Uses bfloat16 precision on CUDA devices for optimal performance
- **Base64 Output**: Returns images as base64-encoded strings for easy API integration

## Requirements

- CUDA-compatible GPU (recommended for optimal performance)
- Python 3.8+
- PyTorch
- Diffusers library
- Transformers library
