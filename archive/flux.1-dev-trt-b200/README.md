# Flux 1.0 Dev - TensorRT

This model provides high-quality text-to-image generation using [Flux.1-dev model](https://huggingface.co/black-forest-labs/FLUX.1-dev) optimized with TensorRT for the B200 GPU.

## Model Information

- **Model**: Flux 1.0 Dev (black-forest-labs/FLUX.1-dev)
- **Optimization**: TensorRT 8.6.1
- **Hardware**: NVIDIA B200 GPU
- **Framework**: PyTorch with TensorRT acceleration

## Features

- High-quality image generation from text prompts
- TensorRT optimization for fast inference
- Support for custom image dimensions (must be multiples of 8)
- Configurable denoising steps and guidance scale
- CUDA graph optimization support

## Usage

### Basic Usage

```bash
curl -X POST https://app.baseten.co/models/{MODEL_ID}/predict \
  -H "Authorization: Api-Key API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful landscape with mountains and a lake, photorealistic, high quality",
    "negative_prompt": "blurry, low quality, distorted",
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 30,
    "guidance_scale": 3.5,
    "seed": 42,
    "batch_size": 1,
    "batch_count": 1
  }' | python show.py
```

### Batch Processing

The model supports efficient batch processing for generating multiple images in a single request. You can generate up to 4 images simultaneously with either the same prompt or different prompts for each image.

#### Same Prompt for All Images

```bash
curl -X POST https://app.baseten.co/models/{MODEL_ID}/predict \
  -H "Authorization: Api-Key API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful landscape with mountains and a lake, photorealistic, high quality",
    "negative_prompt": "blurry, low quality, distorted",
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 30,
    "guidance_scale": 3.5,
    "seed": 42,
    "batch_size": 4,
    "batch_count": 1
  }' | python show_batch.py
```

#### Different Prompts for Each Image

```bash
curl -X POST https://app.baseten.co/models/{MODEL_ID}/predict \
  -H "Authorization: Api-Key API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": [
      "A beautiful landscape with mountains and a lake, photorealistic, high quality",
      "A futuristic city skyline at sunset, neon lights, cyberpunk style, high quality",
      "A cute cat sitting in a garden, soft lighting, detailed, high quality",
      "Abstract geometric patterns in vibrant colors, modern art style, high quality"
    ],
    "negative_prompt": [
      "blurry, low quality, distorted",
      "blurry, low quality, distorted",
      "blurry, low quality, distorted",
      "blurry, low quality, distorted"
    ],
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 30,
    "guidance_scale": 3.5,
    "seed": 42,
    "batch_size": 4,
    "batch_count": 1
  }' | python show_batch.py
```

#### Batch Processing Benefits

- **Parallel Processing**: All images in a batch are generated simultaneously, not sequentially
- **Better GPU Utilization**: More efficient use of GPU resources compared to separate requests
- **Faster Total Time**: Generating 4 images in a batch is significantly faster than 4 separate API calls
- **Consistent Parameters**: All images in a batch use the same dimensions, inference steps, and guidance scale

#### Batch Processing Limitations

- **Maximum Batch Size**: Limited to 4 images per batch (MAX_BATCH_SIZE = 4)
- **Prompt Array Length**: When using different prompts, the prompt array length must match the batch_size
- **Memory Requirements**: Larger batches require more GPU memory

#### Displaying Batch Results

Use the included `show_batch.py` script to handle multiple images in the response:

```bash
# The script automatically detects single vs multiple images
curl ... | python show_batch.py
```

The script will:
- Save each image with a unique filename
- Automatically open all generated images
- Print status information about the batch

### Load Test
Before running the load test, update `load_test.py` with your actual endpoint URL and API key. Replace the placeholder values for `api_url` and `api_key` with your deployment's information (lines 43 and 44).

```bash
python load_test.py --save-all-images --use-varied-prompts --concurrent --num-requests 30

🚀 Starting Flux Truss API test...
==================================================
🎨 Using 30 varied prompts for load testing
📝 First 3 prompts to be tested:
   1. a beautiful photograph of Mt. Fuji during cherry blossom, photorealistic, high quality
   2. a majestic dragon soaring through a mystical forest, digital art, detailed
   3. a cozy coffee shop interior with warm lighting, people working on laptops, photorealistic
   ... and 27 more
🚀 Starting concurrent load test with 30 requests, max 5 workers
============================================================
📤 Sending request 1/30: 'a beautiful photograph of Mt. Fuji during cherry b...'
Testing Truss API endpoint with prompt: 'a beautiful photograph of Mt. Fuji during cherry blossom, photorealistic, high quality'

...

============================================================
📊 LOAD TEST SUMMARY
============================================================
Total requests: 30
Successful: 30
Failed: 0
Success rate: 100.0%
Total time: 74.72 seconds
Average request time: 11.63 seconds
Min request time: 2.91 seconds
Max request time: 12.86 seconds
Throughput: 0.40 requests/second

==================================================

💾 Saving 30 successful images...
✅ Successfully saved 30/30 images to './output'
📁 Opened output directory

==================================================
```

### Advanced Usage

```python
{
    "prompt": "A beautiful landscape with mountains and a lake, photorealistic, high quality",
    "prompt2": "Additional prompt for T5 tokenizer",  # Optional, uses prompt if not provided
    "negative_prompt": "blurry, low quality, distorted",  # Optional
    "height": 1024,  # Must be multiple of 8
    "width": 1024,   # Must be multiple of 8
    "denoising_steps": 50,  # Number of denoising steps
    "guidance_scale": 3.5,   # Classifier-free guidance scale (must be > 1)
    "seed": 42,  # Random seed for reproducible results
    "batch_size": 1,  # Number of images to generate
    "batch_count": 1,  # Number of batches
    "num_warmup_runs": 0,  # Number of warmup runs
    "max_sequence_length": 512,  # Max sequence length (up to 512 for flux.1-dev)
    "t5_ws_percentage": None,  # T5 weight streaming percentage
    "transformer_ws_percentage": None  # Transformer weight streaming percentage
}
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string/array | required | Text prompt(s) for image generation. Can be a single string or array of strings for different prompts per batch item |
| `prompt2` | string/array | same as prompt | Additional prompt(s) for T5 tokenizer. Can be a single string or array of strings |
| `negative_prompt` | string/array | "" | Negative prompt(s) to avoid certain elements. Can be a single string or array of strings |
| `height` | int | 1024 | Image height (must be multiple of 8) |
| `width` | int | 1024 | Image width (must be multiple of 8) |
| `denoising_steps` | int | 50 | Number of denoising steps |
| `guidance_scale` | float | 3.5 | Classifier-free guidance scale (> 1) |
| `seed` | int | None | Random seed for reproducibility |
| `batch_size` | int | 1 | Number of images per batch |
| `batch_count` | int | 1 | Number of batches |
| `num_warmup_runs` | int | 0 | Number of warmup runs |
| `max_sequence_length` | int | 512 | Maximum sequence length (≤ 512) |
| `t5_ws_percentage` | int | None | T5 weight streaming percentage |
| `transformer_ws_percentage` | int | None | Transformer weight streaming percentage |

## Response Format

The model returns a JSON response with the following structure:

### Single Image Response

```json
{
    "status": "success",
    "data": "base64_encoded_image",
    "time": 2.34,
    "prompt": "A beautiful landscape with mountains and a lake, photorealistic, high quality",
    "negative_prompt": "blurry, low quality, distorted",
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 30,
    "guidance_scale": 3.5,
    "seed": 42
}
```

### Batch Response (Multiple Images)

```json
{
    "status": "success",
    "data": [
        "base64_encoded_image_1",
        "base64_encoded_image_2",
        "base64_encoded_image_3",
        "base64_encoded_image_4"
    ],
    "time": 7.23,
    "prompt": [
        "A beautiful landscape with mountains and a lake, photorealistic, high quality",
        "A futuristic city skyline at sunset, neon lights, cyberpunk style, high quality",
        "A cute cat sitting in a garden, soft lighting, detailed, high quality",
        "Abstract geometric patterns in vibrant colors, modern art style, high quality"
    ],
    "negative_prompt": [
        "blurry, low quality, distorted",
        "blurry, low quality, distorted",
        "blurry, low quality, distorted",
        "blurry, low quality, distorted"
    ],
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 30,
    "guidance_scale": 3.5,
    "seed": 42
}
```

## Performance Notes

The model is optimized with a pre-compiled TensorRT engine for the NVIDIA B200 GPU.
Performance characteristic is described below for the [basic usage](#basic-usage)


```text
|------------------|--------------|
|     Module       |   Latency    |
|------------------|--------------|
|      CLIP        |      2.02 ms |
|       T5         |      6.43 ms |
| Transformer x 50 |   2361.44 ms |
|     VAE-Dec      |     11.67 ms |
|------------------|--------------|
|    Pipeline      |   2382.45 ms |
|------------------|--------------|
```

## Model Variants

This implementation supports the `flux.1-dev` variant with:
- Maximum sequence length: 512 tokens
- Default image dimensions: 1024x1024
- Optimized for high-quality image generation
