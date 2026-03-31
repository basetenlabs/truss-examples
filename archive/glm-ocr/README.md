# GLM-OCR Truss Model

This is a Truss deployment of the [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) model for optical character recognition using vLLM engine on Baseten served on an L4 GPU. With only 0.9B parameters, GLM-OCR delivers strong OCR performance while being lightweight enough for high-concurrency and edge deployments.

GLM-OCR integrates the CogViT visual encoder, a lightweight cross-modal connector with efficient token downsampling, and a GLM-0.5B language decoder. It supports a two-stage pipeline (layout analysis + parallel recognition) for processing complex documents.

## Quick Start

### 1. Deploy to Baseten

```bash
# Clone this repo and cd into this folder
git clone https://github.com/basetenlabs/truss-examples.git
cd truss-examples/glm-ocr

# Deploy the model
truss push --publish
# This assumes you have truss installed, if not follow the instructions here:
# https://docs.baseten.co/development/model/build-your-first-model
```

## Model Information

- **Model**: [zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR)
- **Parameters**: 0.9B
- **Framework**: vLLM (OpenAI-compatible API)
- **GPU**: L4 (24GB)
- **API**: OpenAI Chat Completions (`/v1/chat/completions`)

## Usage

### Using the OpenAI Client

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_BASETEN_API_KEY",
    base_url="https://model-XXXX.api.baseten.co/deployment/YYYY/sync/v1"
)

response = client.chat.completions.create(
    model="zai-org/GLM-OCR",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="}},
            {"type": "text", "text": "Text Recognition:"}
        ]
    }],
)

print(response.choices[0].message.content)
```

The model accepts images via URL or base64-encoded data URIs, and returns recognized text in markdown format.
