# NVIDIA Llama NemoRetriever ColEmbed 3B V1

This is a traditional [Truss](https://truss.baseten.co/) implementation for NVIDIA's Llama NemoRetriever ColEmbed 3B V1 cross-modal embedding model. This implementation uses the standard `load()` and `predict()` pattern with a `model.py` file.

## Model Description

The NVIDIA Llama NemoRetriever ColEmbed 3B V1 is a state-of-the-art **cross-modal** embedding model designed for retrieval tasks. It uses a columnar embedding approach (ColEmbed) to generate high-quality embeddings for:

- **Cross-modal retrieval**: Match text queries to image documents
- **Text-to-text retrieval**: Traditional semantic search
- **Image document embedding**: Encode images as searchable documents
- **RAG applications**: Retrieval-augmented generation with multimodal content

## Deployment

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples.git
cd nemotron/llama-embed-nemotron-8b
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`
3. Retrieve your Hugging Face token from the [settings](https://huggingface.co/settings/tokens).
4. Set your Hugging Face token as a Baseten secret [here](https://app.baseten.co/settings/secrets) with the key `hf_access_token`.

With `llama-embed-nemotron-8b` as your working directory, you can deploy the model with:

```sh
truss push --publish
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

## API Documentation

### Request Format

The API accepts a simple JSON payload with the following fields:

```json
{
  "queries": ["Text query 1", "Text query 2"],
  "passages": ["Text passage 1", {"type": "image", "url": "https://..."}],
  "batch_size": 8,
  "compute_scores": false
}
```

**Parameters:**

- **`queries`** (List[str], optional): List of text queries to encode
- **`passages`** (List[str or Dict], optional): List of passages to encode. Can be:
  - Text strings: `"Your text passage here"`
  - Image objects: `{"type": "image", "url": "https://..."}`
  - Image objects: `{"type": "image", "content": "base64_string"}`
- **`batch_size`** (int, optional): Batch size for encoding. Default: `8`
- **`compute_scores`** (bool, optional): Whether to compute similarity scores between queries and passages. Default: `false`

**Notes:**
- You can provide just `queries`, just `passages`, or both
- If both are provided and `compute_scores=true`, you'll get a similarity score matrix

### Response Format

```json
{
  "query_embeddings": [[0.1, 0.2, ...], ...],
  "passage_embeddings": [[0.3, 0.4, ...], ...],
  "scores": [[13.99, 11.42, ...], ...]
}
```

**Fields:**
- **`query_embeddings`**: List of query embeddings (present if queries were provided)
- **`passage_embeddings`**: List of passage embeddings (present if passages were provided)
- **`scores`**: Similarity score matrix (present if `compute_scores=true`)

## Usage Examples

### Example 1: Cross-Modal Retrieval (Text Queries → Image Documents)

This is the primary use case - searching image documents with text queries.

```python
import requests
import os

model_id = "YOUR_MODEL_ID"
api_key = os.environ["BASETEN_API_KEY"]

# Text queries
queries = [
    'How much percentage of Germanys population died in the 2nd World War?',
    'How many million tons CO2 were captured from Gas processing in 2018?',
    'What is the average CO2 emission of someone in Japan?'
]

# Image URLs (documents to search)
image_docs = [
    {
        "type": "image",
        "url": "https://upload.wikimedia.org/wikipedia/commons/3/35/Human_losses_of_world_war_two_by_country.png"
    },
    {
        "type": "image",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/76/20210413_Carbon_capture_and_storage_-_CCS_-_proposed_vs_implemented.svg/2560px-20210413_Carbon_capture_and_storage_-_CCS_-_proposed_vs_implemented.svg.png"
    },
    {
        "type": "image",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/20210626_Variwide_chart_of_greenhouse_gas_emissions_per_capita_by_country.svg/2880px-20210626_Variwide_chart_of_greenhouse_gas_emissions_per_capita_by_country.svg.png"
    }
]

response = requests.post(
    f"https://model-{model_id}.api.baseten.co/production/predict",
    headers={"Authorization": f"Api-Key {api_key}"},
    json={
        "queries": queries,
        "passages": image_docs,
        "batch_size": 8,
        "compute_scores": True
    }
)

result = response.json()
scores = result["scores"]

# scores[i][j] = similarity between query i and image j
# Diagonal should have high scores (matching pairs)
print("Similarity scores:")
for i, query in enumerate(queries):
    print(f"\nQuery {i+1}: {query[:50]}...")
    for j, score in enumerate(scores[i]):
        print(f"  Image {j+1}: {score:.4f}")
```

### Example 2: Text-to-Text Retrieval

```python
import requests
import os

model_id = "YOUR_MODEL_ID"
api_key = os.environ["BASETEN_API_KEY"]

query = "What is deep learning?"
documents = [
    "Deep learning uses neural networks with multiple layers.",
    "The weather is nice today.",
    "Neural networks are inspired by the human brain."
]

response = requests.post(
    f"https://model-{model_id}.api.baseten.co/production/predict",
    headers={"Authorization": f"Api-Key {api_key}"},
    json={
        "queries": [query],
        "passages": documents,
        "batch_size": 8,
        "compute_scores": True
    }
)

result = response.json()
scores = result["scores"][0]  # Get scores for first query

# Rank documents by relevance
ranked = sorted(zip(scores, documents), reverse=True)
print(f"Query: {query}\n")
for i, (score, doc) in enumerate(ranked, 1):
    print(f"{i}. [{score:.4f}] {doc}")
```

### Example 3: Generate Embeddings Only (No Scoring)

Encode queries and passages separately without computing similarity scores.

```python
import requests
import os

model_id = "YOUR_MODEL_ID"
api_key = os.environ["BASETEN_API_KEY"]

# Just encode queries
response = requests.post(
    f"https://model-{model_id}.api.baseten.co/production/predict",
    headers={"Authorization": f"Api-Key {api_key}"},
    json={
        "queries": ["What is AI?", "Explain machine learning"],
        "batch_size": 8
    }
)

result = response.json()
query_embeddings = result["query_embeddings"]
print(f"Generated {len(query_embeddings)} query embeddings")

# Just encode passages (text and images)
response = requests.post(
    f"https://model-{model_id}.api.baseten.co/production/predict",
    headers={"Authorization": f"Api-Key {api_key}"},
    json={
        "passages": [
            "AI is artificial intelligence.",
            {"type": "image", "url": "https://example.com/diagram.png"}
        ],
        "batch_size": 8
    }
)

result = response.json()
passage_embeddings = result["passage_embeddings"]
print(f"Generated {len(passage_embeddings)} passage embeddings")
```

### Example 4: Mixed Text and Image Passages

```python
import requests
import os

model_id = "YOUR_MODEL_ID"
api_key = os.environ["BASETEN_API_KEY"]

query = "What is climate change?"
passages = [
    "Climate change refers to long-term shifts in temperatures and weather patterns.",
    {"type": "image", "url": "https://example.com/climate-chart.png"},
    "Global warming is primarily caused by greenhouse gas emissions."
]

response = requests.post(
    f"https://model-{model_id}.api.baseten.co/production/predict",
    headers={"Authorization": f"Api-Key {api_key}"},
    json={
        "queries": [query],
        "passages": passages,
        "batch_size": 8,
        "compute_scores": True
    }
)

result = response.json()
scores = result["scores"][0]

# Show relevance scores
for i, (score, passage) in enumerate(zip(scores, passages), 1):
    if isinstance(passage, dict):
        print(f"{i}. [{score:.4f}] Image: {passage['url']}")
    else:
        print(f"{i}. [{score:.4f}] Text: {passage[:60]}...")
```

### Example 5: Using curl

```bash
curl -X POST https://model-YOUR_MODEL_ID.api.baseten.co/production/predict \
  -H "Authorization: Api-Key YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": ["What is AI?"],
    "passages": ["Artificial intelligence is the simulation of human intelligence."],
    "batch_size": 8,
    "compute_scores": true
  }'
```

### Example 6: Base64 Image Input

```python
import requests
import base64
import os

model_id = "YOUR_MODEL_ID"
api_key = os.environ["BASETEN_API_KEY"]

# Read and encode image
with open("image.png", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

response = requests.post(
    f"https://model-{model_id}.api.baseten.co/production/predict",
    headers={"Authorization": f"Api-Key {api_key}"},
    json={
        "queries": ["What does this image show?"],
        "passages": [
            {"type": "image", "content": image_base64}
        ],
        "batch_size": 8,
        "compute_scores": True
    }
)

result = response.json()
score = result["scores"][0][0]
print(f"Similarity score: {score:.4f}")
```

## Model Configuration

- **Model Size**: 3B parameters
- **Context Length**: Supports long documents
- **Precision**: BFloat16
- **Hardware**: Optimized for NVIDIA L4, A10G, A100, or H100 GPUs
- **Flash Attention**: Uses Flash Attention 2 for efficiency
- **Model Version**: Pinned to revision `50c36f4d5271c6851aa08bd26d69f6e7ca8b870c`

## Implementation Details

This Truss implementation uses the model's native API methods:

- **`forward_queries(queries, batch_size)`**: Encodes text queries into embeddings
- **`forward_passages(passages, batch_size)`**: Encodes text or image passages into embeddings
- **`get_scores(query_embeddings, passage_embeddings)`**: Computes similarity scores

The model architecture:
- Uses ColBERT-style multi-vector representations
- Implements Flash Attention 2 for memory efficiency
- Supports cross-modal retrieval between text and images

## Performance

On L4 GPU:
- **Query Encoding**: ~50-100ms per batch (8 queries)
- **Passage Encoding**: ~100-200ms per batch (8 passages with images)
- **Throughput**: ~100-300 embeddings/second
- **Concurrency**: Up to 16 concurrent requests
- **Memory Usage**: ~10-12GB GPU memory

## Use Cases

### 1. Visual Question Answering
Search through charts, infographics, and diagrams using natural language queries.

```python
query = "What were the CO2 emissions in 2020?"
image_docs = [list of chart images]
# Returns most relevant chart
```

### 2. Multimodal Document Retrieval
Build a search engine that works across both text documents and images.

```python
query = "Explain neural networks"
docs = ["text doc 1", {"type": "image", "url": "diagram.png"}, "text doc 2"]
# Ranks all documents by relevance
```

### 3. Cross-Modal RAG
Retrieve relevant images and text for LLM context.

```python
query = "Show me data about climate change"
mixed_docs = [text_docs + image_docs]
# Feed top results to LLM as context
```

### 4. Image Search with Text
Find images in a database using text descriptions.

```python
query = "red sports car"
image_database = [list of product images]
# Returns most matching images
```

## Common Patterns

### Pattern 1: Two-Stage Retrieval

First generate embeddings, store them, then search later.

```python
# Stage 1: Index documents (run once)
response = requests.post(url, json={
    "passages": all_documents
})
embeddings = response.json()["passage_embeddings"]
# Store embeddings in vector database

# Stage 2: Search (run many times)
response = requests.post(url, json={
    "queries": [user_query]
})
query_emb = response.json()["query_embeddings"][0]
# Search vector database for similar embeddings
```

### Pattern 2: Batch Processing

Process large datasets efficiently.

```python
# Process in batches
all_queries = [...]  # 1000 queries
batch_size = 32

for i in range(0, len(all_queries), batch_size):
    batch = all_queries[i:i+batch_size]
    response = requests.post(url, json={
        "queries": batch,
        "batch_size": 8
    })
    # Process results
```

### Pattern 3: Reranking

Use the model to rerank initial search results.

```python
# Get initial candidates from keyword search
candidates = get_initial_results(query)

# Rerank with semantic similarity
response = requests.post(url, json={
    "queries": [query],
    "passages": candidates,
    "compute_scores": True
})

scores = response.json()["scores"][0]
reranked = sorted(zip(scores, candidates), reverse=True)
```

## Troubleshooting

### Large Image Files

If images are very large, consider resizing before sending:

```python
from PIL import Image
import io

img = Image.open("large_image.png")
img.thumbnail((1024, 1024))  # Resize
# Convert to base64 and send
```

### Timeout Errors

For many images or slow networks, increase timeout:

```python
response = requests.post(url, json={...}, timeout=60)
```

### Memory Issues

If you hit memory limits, reduce batch size:

```python
{
    "queries": [...],
    "passages": [...],
    "batch_size": 4  # Smaller batches
}
```

## Support

For questions or issues, please refer to:
- [Truss Documentation](https://docs.baseten.co)
- [Model Card](https://huggingface.co/nvidia/llama-nemoretriever-colembed-3b-v1)
