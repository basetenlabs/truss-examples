
# Performance Section
Below are a example deployments of optimized models for the Baseten platform.

# Baseten Embeddings Inference (BEI)

Collection of BEI (Baseten Embeddings Inference) model implementations for deployment to Baseten. BEI is Baseten's solution for production-grade embeddings/re-ranking and classification inference using TensorRT-LLM.

With BEI you get the following benefits:
- *Lowest-latency inference* across any embedding solution (vLLM, SGlang, Infinity, TEI, Ollama)<sup>1</sup>
- *Highest-throughput inference* across any embedding solution (vLLM, SGlang, Infinity, TEI, Ollama) - thanks to XQA kernels, FP8 and dynamic batching.<sup>2</sup>
- High parallelism: up to 1400 client embeddings per second
- Cached model weights for fast vertical scaling and high availability - no Hugging Face hub dependency at runtime

Architectures that are not supported on BEI are deployed with Huggingface's text-embeddings-inference (TEI) solution.

# Examples:
You can find the following deployments in this repository:

## Embedding Model Deployments:
 - [Alibaba-NLP/gte-Qwen2-1.5B-instruct-embedding-TEI](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/TEI-alibaba-nlp-gte-qwen2-1.5b-instruct-embedding)
 - [BAAI/bge-en-icl-embedding-BEI](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/BEI-baai-bge-en-icl-embedding)
 - [BAAI/bge-large-en-v1.5-embedding-BEI](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/BEI-baai-bge-large-en-v1.5-embedding)
 - [BAAI/bge-m3-embedding-dense-BEI](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/BEI-baai-bge-m3-embedding-dense)
 - [BAAI/bge-multilingual-gemma2-multilingual-embedding-BEI](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/BEI-baai-bge-multilingual-gemma2-multilingual-embedding)
 - [Salesforce/SFR-Embedding-Mistral-BEI](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/BEI-salesforce-sfr-embedding-mistral)
 - [Snowflake/snowflake-arctic-embed-l-v2.0-BEI](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/BEI-snowflake-snowflake-arctic-embed-l-v2.0)
 - [WhereIsAI/UAE-Large-V1-embedding-BEI](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/BEI-whereisai-uae-large-v1-embedding)
 - [intfloat/e5-mistral-7b-instruct-embedding-BEI](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/BEI-intfloat-e5-mistral-7b-instruct-embedding)
 - [intfloat/multilingual-e5-large-instruct-TEI](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/TEI-intfloat-multilingual-e5-large-instruct)
 - [jina-ai/jina-embeddings-v2-base-en-TEI](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/TEI-jina-ai-jina-embeddings-v2-base-en)
 - [jinaai/jina-embeddings-v2-base-code-TEI](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/TEI-jinaai-jina-embeddings-v2-base-code)
 - [mixedbread-ai/mxbai-embed-large-v1-embedding-BEI](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/BEI-mixedbread-ai-mxbai-embed-large-v1-embedding)
 - [nomic-ai/nomic-embed-text-v1.5-TEI](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/TEI-nomic-ai-nomic-embed-text-v1.5)
 - [sentence-transformers/all-MiniLM-L6-v2-embedding-TEI](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/TEI-sentence-transformers-all-minilm-l6-v2-embedding)

## Reranker Deployments:
 - [Alibaba-NLP/gte-multilingual-reranker-base-TEI](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/TEI-alibaba-nlp-gte-multilingual-reranker-base)
 - [BAAI/bge-reranker-large-BEI](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/BEI-baai-bge-reranker-large)
 - [BAAI/bge-reranker-v2-m3-multilingual-BEI](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/BEI-baai-bge-reranker-v2-m3-multilingual)
 - [ncbi/MedCPT-Cross-Encoder-reranker-BEI](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/BEI-ncbi-medcpt-cross-encoder-reranker)

## Text Sequence Classification Deployments:
 - [SamLowe/roberta-base-go_emotions-classification-BEI](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/BEI-samlowe-roberta-base-go_emotions-classification)
 - [Skywork/Skywork-Reward-Llama-3.1-8B-v0.2-Reward-Model-BEI](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/BEI-skywork-skywork-reward-llama-3.1-8b-v0.2-reward-model)
 - [allenai/Llama-3.1-Tulu-3-8B-Reward-Model-BEI](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/BEI-allenai-llama-3.1-tulu-3-8b-reward-model)
 - [baseten/example-Meta-Llama-3-70B-InstructForSequenceClassification-BEI](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/BEI-baseten-example-meta-llama-3-70b-instructforsequenceclassification)
 - [papluca/xlm-roberta-base-language-detection-classification-BEI](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/BEI-papluca-xlm-roberta-base-language-detection-classification)

<sup>1</sup> measured on H100-HBM3 (bert-large-335M, for MistralModel-7B: 9ms)
<sup>2</sup> measured on H100-HBM3 (leading model architecture on MTEB, MistralModel-7B)
