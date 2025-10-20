# DeepSeek OCR Truss Model

This is a Truss deployment of the DeepSeek OCR model for optical character recognition using vLLM engine.

## Model Information

- **Model**: DeepSeek OCR v1
- **Framework**: vLLM + PyTorch + Transformers
- **GPU**: A10G recommended
- **Memory**: 16GB RAM
- **Engine**: AsyncLLMEngine with custom DeepseekOCRForCausalLM

## Features

- **Advanced OCR**: Extracts text with bounding box coordinates
- **Reference Processing**: Handles `<|ref|>` and `<|det|>` tokens for structured output
- **Image Processing**: Automatic resizing, padding, and EXIF correction
- **N-gram Prevention**: Prevents repetitive text generation
- **Multiple Input Formats**: URL, base64, or direct image data
- **Separate Visualization Module**: `visualizer.py` for bounding box and geometric visualization
- **Clean API**: Model returns OCR results, visualization handled separately

## Usage

### Input Format

The model accepts the following input formats:

```json
{
  "image_url": "https://example.com/image.jpg",
  "prompt": "Extract all text from this image. <image>"
}
```

Or with base64 encoded image:

```json
{
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "prompt": "Extract all text from this image. <image>"
}
```

### Output Format

For images with references:
```json
{
  "extracted_text": "Cleaned text with image references",
  "raw_output": "Raw model output with <|ref|> tokens",
  "references": [["<|ref|>text<|/ref|><|det|>coords<|/det|>", ...]],
  "image_references": ["<|ref|>image<|/ref|><|det|>coords<|/det|>", ...],
  "other_references": ["<|ref|>other<|/ref|><|det|>coords<|/det|>", ...],
  "image_size": [width, height],
  "prompt_used": "Extract all text from this image. <image>",
  "model_name": "deepseek-ai/DeepSeek-OCR",
  "has_bounding_boxes": true
}
```

For simple text extraction:
```json
{
  "extracted_text": "The extracted text from the image",
  "image_size": [width, height],
  "prompt_used": "Extract all text from this image. <image>",
  "model_name": "deepseek-ai/deepseek-ocr-v1"
}
```

## Deployment

1. Deploy using Truss:
```bash
truss push
```

2. Test the deployment:
```bash
truss predict -d '{"image_url": "https://example.com/image.jpg", "prompt": "Extract all text from this image. <image>"}'
```

3. Create visualizations (optional):
```python
from visualizer import DeepSeekOCRVisualizer
from PIL import Image

# Load image and OCR result
image = Image.open("your_image.jpg")
ocr_result = {"raw_output": "your_ocr_output", "has_bounding_boxes": True}

# Create visualization
visualizer = DeepSeekOCRVisualizer()
viz_result = visualizer.create_visualization(image, ocr_result)

# Save visualization
if viz_result:
    import base64
    viz_image = Image.open(io.BytesIO(base64.b64decode(viz_result['visualization'])))
    viz_image.save("visualization.png")
```

## Configuration

The model uses several configuration parameters:

- **Model Path**: `deepseek-ai/deepseek-ocr-v1`
- **Max Model Length**: 8192 tokens
- **GPU Memory Utilization**: 75%
- **N-gram Size**: 30 (for repetition prevention)
- **Window Size**: 90 (for sliding window)

## Dependencies

- `vllm==0.8.5` - vLLM inference engine
- `torch==2.6.0` - PyTorch framework
- `transformers==4.51.1` - Hugging Face transformers
- `flash-attn==2.7.3` - Flash attention implementation
- `pillow==10.0.0` - Image processing
- `numpy==1.24.3` - Numerical computations
- `matplotlib==3.7.2` - Plotting (for geometric outputs)

## Build Commands

The DeepSeek-OCR repository is cloned during the build stage using Baseten's build commands:

```yaml
build_commands:
  - git clone https://github.com/deepseek-ai/DeepSeek-OCR.git
  - cd DeepSeek-OCR && pip install -r requirements.txt
```

This approach provides:
- **Build-time caching** of the repository and dependencies
- **Reduced cold starts** by pre-installing everything
- **Reproducible builds** across deployments

## Notes

- The model uses async generation for better performance
- Custom logits processor prevents n-gram repetition
- Image processing includes automatic resizing and padding
- Reference tokens are processed to extract structured information
- Mock implementation available for testing without GPU
- Repository is cloned during build step using Baseten build commands
- Uses official DeepSeek-OCR implementation with vLLM support
- Build-time caching reduces deployment cold starts
