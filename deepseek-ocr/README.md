# DeepSeek OCR Truss Model

This is a Truss deployment of the DeepSeek OCR model for optical character recognition using vLLM engine. The model excels at reading handwritten text, documents, and complex layouts with bounding box detection.

## Quick Start

### 1. Deploy to Baseten

```bash
# Set your Baseten API key
export BASETEN_API_KEY="your_api_key_here"

# Deploy the model
truss push
```

### 2. Test with Sample Image

```bash
# Run the test script with Bad-Handwriting.png
cd deepseek-ocr
python test_document_ocr.py
```

This will:
- Load `Bad-Handwriting.png` (a challenging handwriting sample)
- Test 5 different OCR prompts
- Generate visualizations with bounding boxes
- Save results as `visualization_*.png` files

## Model Information

- **Model**: DeepSeek OCR v1
- **Framework**: vLLM + PyTorch + Transformers
- **GPU**: H100_40GB recommended
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

### Using the Test Script

The `test_document_ocr.py` script provides a complete example of how to use the model:

```python
# The script tests these prompts:
prompts = [
    "<image>\n<|grounding|>Convert the document to markdown.",
    "<image>\n<|grounding|>OCR this image.",
    "<image>\nFree OCR.",
    "<image>\nParse the figure.",
    "<image>\nDescribe this image in detail.",
]
```

### API Input Format

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

### Recommended Prompts

For best results, use these prompts:

1. **`<image>\n<|grounding|>Convert the document to markdown.`** - Best for structured documents with bounding boxes
2. **`<image>\nFree OCR.`** - Good for simple text extraction without bounding boxes
3. **`<image>\n<|grounding|>OCR this image.`** - Detailed detection but may fragment text

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
  "model_name": "deepseek-ai/DeepSeek-OCR"
}
```

## Visualization

The model includes an enhanced visualizer with smart label spacing to prevent overlap:

```python
from visualizer import DeepSeekOCRVisualizer
from PIL import Image
import base64
import io

# Load image and OCR result
image = Image.open("your_image.jpg")
ocr_result = {
    "raw_output": "your_ocr_output_with_ref_tokens",
    "has_bounding_boxes": True
}

# Create visualization
visualizer = DeepSeekOCRVisualizer()
viz_result = visualizer.create_visualization(image, ocr_result)

# Save visualization
if viz_result:
    viz_image = Image.open(io.BytesIO(base64.b64decode(viz_result['visualization'])))
    viz_image.save("visualization.png")
    print(f"Has bounding boxes: {viz_result['has_bounding_boxes']}")
```

### Visualization Features

- **Smart Label Spacing**: Automatically prevents label overlap
- **Bounding Box Detection**: Shows detected text regions
- **Color-coded Regions**: Different colors for titles, text, and other elements
- **Geometric Visualization**: Optional matplotlib-based geometric elements

## Configuration

The model uses several configuration parameters:

- **Model Path**: `deepseek-ai/DeepSeek-OCR`
- **Max Model Length**: 8192 tokens
- **GPU Memory Utilization**: 75%
- **N-gram Size**: 30 (for repetition prevention)
- **Window Size**: 90 (for sliding window)
- **Model Implementation**: `transformers` (forced for compatibility)

## Dependencies

- `vllm==0.8.5` - vLLM inference engine
- `transformers==4.46.3` - Hugging Face transformers
- `tokenizers==0.20.3` - Tokenization library
- `PyMuPDF` - PDF processing
- `img2pdf` - Image to PDF conversion
- `einops` - Tensor operations
- `easydict` - Dictionary utilities
- `addict` - Dictionary enhancements
- `Pillow` - Image processing
- `numpy` - Numerical computations

## Build Commands

The DeepSeek-OCR repository is cloned during the build stage:

```yaml
build_commands:
  - apt-get update && apt-get install -y python3-dev
  - git clone https://github.com/deepseek-ai/DeepSeek-OCR.git /DeepSeek-OCR
  - pip install -r /DeepSeek-OCR/requirements.txt
  - pip install vllm==0.8.5
  - pip install flash-attn==2.7.3 --no-build-isolation
```

## Troubleshooting

### Common Issues

1. **Import Errors**: The model falls back to mock implementation if DeepSeek-OCR imports fail
2. **Version Compatibility**: Uses `transformers==4.46.3` with vLLM 0.8.5
3. **Label Overlap**: The visualizer automatically handles label spacing to prevent overlap
4. **Empty Text**: Some prompts may detect bounding boxes but extract minimal text

### Debug Mode

Enable debug logging by checking the model output:

```python
# Check raw output for debugging
raw_output = result.get("raw_output", "")
if raw_output:
    print(f"Raw output: {raw_output[:500]}...")
```

## Notes

- The model uses async generation for better performance
- Custom logits processor prevents n-gram repetition
- Image processing includes automatic resizing and padding
- Reference tokens are processed to extract structured information
- Mock implementation available for testing without GPU
- Enhanced visualizer with smart label spacing
- Compatible with Baseten's CUDA Python base image
