# Kaiko Midnight - Pathology Foundation Model

A Baseten deployment of the [Kaiko Midnight](https://huggingface.co/kaiko-ai/midnight) pathology foundation model for medical image analysis and classification.

## Overview

Kaiko Midnight is a 1.14B parameter pathology foundation model based on DINOv2, optimized for medical image analysis. It provides both classification and segmentation embeddings for pathology images.

## Deploy Kiako Midnight
First, clone this repository:

```bash
git clone https://github.com/basetenlabs/truss-examples/
cd midnight
```

Before deployment:

Make sure you have a Baseten account and API key.
Install the latest version of Truss: `pip install --upgrade truss`
With midnight as your working directory, you can deploy the model with:

```bash
truss push
```

Paste your Baseten API key if prompted.

For more information, see Truss documentation.

Once your Truss is deployed, you can start using Midnight through the Baseten platform! Navigate to the Baseten UI to watch the model build and deploy and invoke it via the REST API.

Note: If you run into the following error during the build phase, downgrade truss with: `pip install truss==0.9.111`

```
error: failed to solve: process "/bin/sh -c uv pip install --python $(which python3) -r base_server_requirements.txt --no-cache-dir" did not complete successfully: exit code: 2
```

## GPU Requirements

- **Minimum**: T4 (16GB VRAM)

## API Usage

#### Single Image Processing
```json
{
  "image_url": "https://upload.wikimedia.org/wikipedia/commons/8/80/Breast_DCIS_histopathology_%281%29.jpg",
  "task": "classification",
  "batch_size": 1
}
```

#### Single Image with Base64
```json
{
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "task": "classification",
  "batch_size": 1
}
```

#### Batch Processing (Multiple Images)
```json
{
  "image_urls": [
    "https://example.com/image1.jpg",
    "https://example.com/image2.jpg",
    "https://example.com/image3.jpg",
    "https://example.com/image4.jpg"
  ],
  "task": "classification",
  "batch_size": 4
}
```

#### True Batch Processing with Base64
```json
{
  "image_base64_list": [
    "iVBORw0KGgoAAAANSUhEUgAA...",
    "iVBORw0KGgoAAAANSUhEUgAA...",
    "iVBORw0KGgoAAAANSUhEUgAA...",
    "iVBORw0KGgoAAAANSUhEUgAA..."
  ],
  "task": "classification",
  "batch_size": 4
}
```

### Response Format
```json
{
  "embeddings": [[0.123, 0.456, ...], [0.789, 0.012, ...]],
  "embedding_shape": [2, 3072],
  "task": "classification",
  "model_id": "kaiko-ai/midnight",
  "input_size": 224,
  "actual_batch_size": 2,
  "requested_batch_size": 4,
  "optimal_batch_size": 8,
  "gpu_memory_gb": 16.0
}
```

## Example Usage

### Python Client - Single Image
```python
import requests
import base64

# Baseten endpoint URL
endpoint_url = "https://your-baseten-endpoint.baseten.co"

# Example request for classification
request_data = {
    "image_url": "https://upload.wikimedia.org/wikipedia/commons/8/80/Breast_DCIS_histopathology_%281%29.jpg",
    "task": "classification",
    "batch_size": 1
}

# Make prediction
response = requests.post(
    endpoint_url,
    headers={"Authorization": "Api-Key YOUR_API_KEY"},
    json=request_data
)
result = response.json()

print(f"Task: {result['task']}")
print(f"Embedding shape: {result['embedding_shape']}")
print(f"Embedding (first 10 values): {result['embeddings'][0][:10]}")
```

### Python Client - True Batch Processing
```python
import requests
import base64

def encode_image_to_base64(image_path):
    """Convert local image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Baseten endpoint URL
endpoint_url = "https://your-baseten-endpoint.baseten.co"

# Encode multiple local images
image_paths = ["path/to/image1.jpg", "path/to/image2.jpg", "path/to/image3.jpg", "path/to/image4.jpg"]
image_base64_list = [encode_image_to_base64(path) for path in image_paths]

# Example request for batch classification
request_data = {
    "image_base64_list": image_base64_list,
    "task": "classification",
    "batch_size": 4
}

# Make prediction
response = requests.post(
    endpoint_url,
    headers={"Authorization": "Api-Key YOUR_API_KEY"},
    json=request_data
)
result = response.json()

print(f"Task: {result['task']}")
print(f"Embedding shape: {result['embedding_shape']}")
print(f"Actual batch size: {result['actual_batch_size']}")
print(f"Number of embeddings: {len(result['embeddings'])}")
```

### cURL Example - Single Image
```bash
curl -X POST "https://your-baseten-endpoint.baseten.co" \
  -H "Authorization: Api-Key YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://upload.wikimedia.org/wikipedia/commons/8/80/Breast_DCIS_histopathology_%281%29.jpg",
    "task": "classification",
    "batch_size": 1
  }'
```

### cURL Example - True Batch Processing
```bash
# First, encode your images to base64
IMAGE1_BASE64=$(base64 -i path/to/image1.jpg)
IMAGE2_BASE64=$(base64 -i path/to/image2.jpg)
IMAGE3_BASE64=$(base64 -i path/to/image3.jpg)
IMAGE4_BASE64=$(base64 -i path/to/image4.jpg)

# Then make the batch request
curl -X POST "https://your-baseten-endpoint.baseten.co" \
  -H "Authorization: Api-Key YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"image_base64_list\": [\"$IMAGE1_BASE64\", \"$IMAGE2_BASE64\", \"$IMAGE3_BASE64\", \"$IMAGE4_BASE64\"],
    \"task\": \"classification\",
    \"batch_size\": 4
  }"
```

## Input Requirements

### Image Requirements
- **Format**: JPEG, PNG, TIFF, or any format supported by PIL
- **Size**: Any size (automatically resized to 224x224)
- **Channels**: RGB (automatically converted if needed)
- **Max Size**: Subject to Baseten's request size limits (typically 10MB)

### Base64 Requirements
- **Encoding**: Standard base64 encoding
- **Data URL support**: Both raw base64 and data URLs (data:image/...) are supported
- **Size limit**: Subject to Baseten's request size limits (typically 10MB)
- **Format**: Any image format supported by PIL

### Batch Processing Requirements
- **True Batch**: Process multiple different images in a single request
- **Batch Size**: Automatically optimized based on GPU memory
- **Image Count**: Should match or exceed requested batch_size for optimal performance
- **Memory Efficiency**: Better GPU utilization than single image processing

## Task Types

### Classification
- **Purpose**: Global image embeddings for classification tasks
- **Output**: Concatenated CLS token + mean patch embeddings
- **Shape**: `(batch_size, 3072)` - 1536 + 1536 dimensions
- **Use Cases**: Image classification, similarity search, feature extraction

### Segmentation
- **Purpose**: Spatial embeddings for segmentation tasks
- **Output**: Patch embeddings reshaped to spatial dimensions
- **Shape**: `(batch_size, 1536, 16, 16)` - 16x16 spatial grid
- **Use Cases**: Semantic segmentation, object detection, spatial analysis



## Model Architecture

### Base Model
- **Architecture**: DINOv2 Vision Transformer
- **Parameters**: 1.14B parameters
- **Precision**: F32 (float32)
- **Input Size**: 224x224 pixels
- **Patch Size**: 14x14 pixels
- **Hidden Size**: 1536 dimensions

### Embedding Extraction
- **Classification**: CLS token + mean patch embeddings (3072 dimensions)
- **Segmentation**: Patch embeddings reshaped to spatial grid (1536x16x16)
- **Normalization**: Mean=(0.5,0.5,0.5), Std=(0.5,0.5,0.5)
