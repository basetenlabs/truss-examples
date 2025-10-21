# DeepSeek-OCR Truss Model

This is a Truss deployment of the DeepSeek OCR model for optical character recognition using vLLM engine on Baseten served on a H100_40G. The model excels at reading handwritten text, documents, and complex layouts with bounding box detection.

DeepSeek-OCR processes 200k+ pages/day on a single GPU or 33M pages/day on 20 nodes. It requires 10x fewer visual tokens than text tokens, which means OCR compresses information 10x more efficiently than the text, with decoding precision of 97%. This makes it an excellent model for generating training data as well as potential tasks that involve long-context windows and memory.

## Quick Start

### 1. Deploy to Baseten

```bash
# Set your Baseten API key
export BASETEN_API_KEY="your_api_key_here"

# Clone this repo and cd into this folder
git clone https://github.com/basetenlabs/truss-examples.git
cd truss-examples/deepseek-ocr

# Deploy the model
truss push
```

### 2. Test with Sample Image

```bash
# Run the test script with Bad-Handwriting.png
python test_document_ocr.py
```

This will:
- Load `Bad-Handwriting.png` (a challenging handwriting sample)
- Test 5 different OCR prompts
- Generate visualizations with bounding boxes
- Save results as `visualization_*.png` files

### 3. Project Structure

```
deepseek-ocr/
├── config.yaml              # Truss configuration
├── model/
│   ├── __init__.py
│   └── model.py            # Main model implementation
├── README.md                # Documentation
├── test_document_ocr.py     # Working test script
├── visualizer.py           # Bounding box visualization
├── Bad-Handwriting.png     # Test image
└── visualization_*.png     # Generated outputs
```

## Model Information

- **Model**: DeepSeek OCR v1
- **Framework**: vLLM + PyTorch + Transformers
- **GPU**: H100_40GB recommended
- **Memory**: 16GB RAM
- **Engine**: AsyncLLMEngine with custom DeepseekOCRForCausalLM

## Usage

### Using the Test Script

The `test_document_ocr.py` script is the main testing interface and provides a complete example of how to use the model:

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

**Recommended**: Use `<image>\n<|grounding|>Convert the document to markdown.` for best results with bounding boxes.

For detailed API documentation and examples, see the `test_document_ocr.py` script.
