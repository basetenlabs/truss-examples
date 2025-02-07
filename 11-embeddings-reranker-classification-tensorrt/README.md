
# BEI with Baseten

This is a collection of BEI deployments with Baseten. BEI is Baseten's solution for production-grade deployments via TensorRT-LLM.

With BEI you get the following benefits:
- *lowest-latency inference* across any embedding solution (vLLM, SGlang, Infinity, TEI, Ollama)*1
- *highest-throughput inference* across any embedding solution (vLLM, SGlang, Infinity, TEI, Ollama) - thanks to XQA kernels, FP8 and dynamic batching.*2 
- high parallelism: up to 1400 client embeddings per second
- cached model weights for fast vertical scaling and high availability - no Hugging Face hub dependency at runtime

# Examples:
You can find the following deployments in this repository:

## Embedding Model Deployments:
 - [BAAI/bge-en-icl-embedding](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/BEI-baai-bge-en-icl-embedding)
 - [BAAI/bge-large-en-v1.5-embedding](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/BEI-baai-bge-large-en-v1.5-embedding)
 - [BAAI/bge-multilingual-gemma2-multilingual-embedding](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/BEI-baai-bge-multilingual-gemma2-multilingual-embedding)
 - [Linq-AI-Research/Linq-Embed-Mistral](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/BEI-linq-ai-research-linq-embed-mistral)
 - [Salesforce/SFR-Embedding-Mistral](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/BEI-salesforce-sfr-embedding-mistral)
 - [Snowflake/snowflake-arctic-embed-l-v2.0](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/BEI-snowflake-snowflake-arctic-embed-l-v2.0)
 - [WhereIsAI/UAE-Large-V1-embedding](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/BEI-whereisai-uae-large-v1-embedding)
 - [intfloat/e5-mistral-7b-instruct-embedding](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/BEI-intfloat-e5-mistral-7b-instruct-embedding)
 - [intfloat/multilingual-e5-large-instruct-embedding](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/BEI-intfloat-multilingual-e5-large-instruct-embedding)

## Reranker Deployments:
 - [BAAI/bge-reranker-large](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/BEI-baai-bge-reranker-large)
 - [cross-encoder/ms-marco-MiniLM-L-6-v2-reranker](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/BEI-cross-encoder-ms-marco-minilm-l-6-v2-reranker)

## Text Sequence Classification Deployments:
 - [ProsusAI/finbert-classification](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/BEI-prosusai-finbert-classification)
 - [SamLowe/roberta-base-go_emotions-classification](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/BEI-samlowe-roberta-base-go_emotions-classification)
 - [Skywork/Skywork-Reward-Llama-3.1-8B-v0.2-Reward-Model](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/BEI-skywork-skywork-reward-llama-3.1-8b-v0.2-reward-model)

* measured on H100-HBM3 (bert-large-335M, for MistralModel-7B: 9ms)
** measured on H100-HBM3 (leading model architecture on MTEB, MistralModel-7B)
