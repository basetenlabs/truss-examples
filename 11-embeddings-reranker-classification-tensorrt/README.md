
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
 - [BAAI/bge-en-icl-embedding-BEI](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/BEI-baai-bge-en-icl-embedding-fp8)
 - [BAAI/bge-large-en-v1.5-embedding-BEI](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/BEI-baai-bge-large-en-v1.5-embedding)
 - [BAAI/bge-m3-embedding-dense-BEI](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/BEI-baai-bge-m3-embedding-dense)
 - [BAAI/bge-multilingual-gemma2-multilingual-embedding-BEI](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/BEI-baai-bge-multilingual-gemma2-multilingual-embedding)
 - [Salesforce/SFR-Embedding-Mistral-BEI](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/BEI-salesforce-sfr-embedding-mistral-fp8)
 - [Snowflake/snowflake-arctic-embed-l-v2.0-BEI](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/BEI-snowflake-snowflake-arctic-embed-l-v2.0)
 - [WhereIsAI/UAE-Large-V1-embedding-BEI](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/BEI-whereisai-uae-large-v1-embedding)
 - [intfloat/e5-mistral-7b-instruct-embedding-BEI](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/BEI-intfloat-e5-mistral-7b-instruct-embedding-fp8)
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
 - [Skywork/Skywork-Reward-Llama-3.1-8B-v0.2-Reward-Model-BEI](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/BEI-skywork-skywork-reward-llama-3.1-8b-v0.2-reward-model-fp8)
 - [Skywork/Skywork-Reward-Llama-3.1-8B-v0.2-Reward-Model-BEI](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/BEI-skywork-skywork-reward-llama-3.1-8b-v0.2-reward-model)
 - [allenai/Llama-3.1-Tulu-3-8B-Reward-Model-BEI](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/BEI-allenai-llama-3.1-tulu-3-8b-reward-model-fp8)
 - [baseten/example-Meta-Llama-3-70B-InstructForSequenceClassification-BEI](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/BEI-baseten-example-meta-llama-3-70b-instructforsequenceclassification-fp8)
 - [papluca/xlm-roberta-base-language-detection-classification-BEI](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/BEI-papluca-xlm-roberta-base-language-detection-classification)

<sup>1</sup> measured on H100-HBM3 (bert-large-335M, for MistralModel-7B: 9ms)
<sup>2</sup> measured on H100-HBM3 (leading model architecture on MTEB, MistralModel-7B)

# Text-Generation - Briton
Briton is Baseten's solution for production-grade deployments via TensorRT-LLM for Text-generation models. (e.g. LLama, Qwen, Mistral)

With Briton you get the following benefits by default:
- *Lowest-latency* latency, beating frameworks such as vllm
- *Highest-throughput* inference - tensorrt-llm will automatically use XQA kernels, paged kv caching and inflight batching.
- *distributed inference* run large models (such as LLama-3-405B) in tensor-parallel
- *json-schema based structured output for any model*
- *chunked prefilling* for long generation tasks

Optionally, you can also enable:
- *speculative decoding* using external draft models or lookahead decoding
- *fp8 quantization* on new GPUS such as H100, H200 and L4 GPUs

Examples:
 - [Qwen/Qwen2.5-72B-Instruct-Briton](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/Briton-qwen-qwen2.5-72b-instruct)
 - [deepseek-ai/DeepSeek-R1-Distill-Llama-70B-Briton](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/Briton-deepseek-ai-deepseek-r1-distill-llama-70b)
 - [deepseek-ai/DeepSeek-R1-Distill-Qwen-32B-Briton](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/Briton-deepseek-ai-deepseek-r1-distill-qwen-32b)
 - [meta-llama/Llama-3.1-405B-Briton](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/Briton-meta-llama-llama-3.1-405b)
 - [meta-llama/Llama-3.2-3B-Instruct-Briton](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/Briton-meta-llama-llama-3.2-3b-instruct)
 - [meta-llama/Llama-3.3-70B-Instruct-Briton](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/Briton-meta-llama-llama-3.3-70b-instruct)
 - [meta-llama/Llama-3.3-70B-Instruct-tp2-Briton](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/Briton-meta-llama-llama-3.3-70b-instruct-tp2)
 - [microsoft/phi-4-Briton](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/Briton-microsoft-phi-4)
 - [mistralai/Mistral-7B-Instruct-v0.3-Briton](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/Briton-mistralai-mistral-7b-instruct-v0.3)
 - [mistralai/Mistral-Small-24B-Instruct-2501-Briton](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/Briton-mistralai-mistral-small-24b-instruct-2501)
 - [tiiuae/Falcon3-10B-Instruct-Briton](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt/Briton-tiiuae-falcon3-10b-instruct)
