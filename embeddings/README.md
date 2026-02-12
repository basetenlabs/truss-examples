# Embeddings

Truss configurations for text embedding models, rerankers, and classifiers. Includes both TensorRT-optimized and HuggingFace TEI-based deployments covering a broad set of embedding providers.

| Directory | Models | Description |
|-----------|--------|-------------|
| [bei](bei/) | 50 | Baseten Embeddings Infrastructure -- TensorRT-optimized embedding, reranking, and classification models from providers including BGE, GTE, Nomic, Jina, Qwen 3, Snowflake, and more |
| [tei](tei/) | 15 | HuggingFace Text Embeddings Inference server configurations for embedding and reranking models |
| [clip](clip/) | 1 | OpenAI CLIP model for image and text embeddings |
| [text-embeddings-inference](text-embeddings-inference/) | 1 | Standalone Text Embeddings Inference server configuration |

## Deploying

Each embedding model can be deployed to Baseten with:

```bash
truss push
```
