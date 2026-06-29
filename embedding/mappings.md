# Config → Template Mappings

Maps every config in `archive/11-embeddings-reranker-classification-tensorrt` to its base template.

---

## `bei-embedder.yaml`
`base-model: encoder` · `/v1/embeddings` · no quantization

| Config folder | Accelerator |
|---|---|
| BEI-baai-bge-large-en-v1.5-embedding | L4 |
| BEI-baai-bge-m3-embedding-dense | H100 |
| BEI-baai-bge-multilingual-gemma2-multilingual-embedding | H100-40GB |
| BEI-mixedbread-ai-mxbai-embed-large-v1-embedding | L4 |
| BEI-snowflake-snowflake-arctic-embed-l-v2.0 | H100 |
| BEI-whereisai-uae-large-v1-embedding | L4 |

---

## `bei-embedder-fp8.yaml`
`base-model: encoder` · `/v1/embeddings` · `quantization-type: fp8`

| Config folder | Accelerator |
|---|---|
| BEI-baai-bge-en-icl-embedding-fp8 | H100 |
| BEI-codefuse-ai-f2llm-4b-embedding-fp8 | H100 |
| BEI-intfloat-e5-mistral-7b-instruct-embedding-fp8 | H100 |
| BEI-jinaai-jina-code-embeddings-0.5b-fp8 | H100-40GB |
| BEI-nomic-ai-nomic-embed-code-fp8 | H100-40GB |
| BEI-qwen-qwen3-embedding-0.6b-fp8 | L4 |
| BEI-qwen-qwen3-embedding-4b-fp8 | H100-40GB |
| BEI-qwen-qwen3-embedding-8b-fp8 | H100-40GB |
| BEI-salesforce-sfr-embedding-mistral-fp8 | H100-40GB |

---

## `bei-embedder-fp4.yaml`
`base-model: encoder` · `/v1/embeddings` · `quantization-type: fp4` · B200 only

| Config folder | Accelerator |
|---|---|
| BEI-qwen-qwen3-embedding-4b-fp4 | B200 |

---

## `bei-reranker.yaml`
`base-model: encoder` · `/rerank` · no quantization

| Config folder | Accelerator |
|---|---|
| BEI-baai-bge-reranker-large | L4 |
| BEI-baai-bge-reranker-v2-m3-multilingual | H100 |
| BEI-ncbi-medcpt-cross-encoder-reranker | A10G |

---

## `bei-predictor.yaml`
`base-model: encoder` · `/predict` · no quantization

| Config folder | Accelerator |
|---|---|
| BEI-papluca-xlm-roberta-base-language-detection-classification | L4 |
| BEI-samlowe-roberta-base-go-emotions-classification | L4 |

---

## `bei-predictor-fp8.yaml`
`base-model: encoder` · `/predict` · `quantization-type: fp8`

| Config folder | Accelerator | Notes |
|---|---|---|
| BEI-allenai-llama-3.1-tulu-3-8b-reward-model-fp8 | H100-40GB | Reward model |
| BEI-baseten-example-meta-llama-3-70b-instructforsequenceclassification-fp8 | H100 | Reward model |
| BEI-mixedbread-ai-mxbai-rerank-base-v2-reranker-fp8 | L4 | Causal reranker via /predict |
| BEI-mixedbread-ai-mxbai-rerank-large-v2-reranker-fp8 | L4 | Causal reranker via /predict |
| BEI-qwen-qwen3-reranker-0.6b-fp8 | L4 | Causal reranker via /predict |
| BEI-qwen-qwen3-reranker-4b-fp8 | H100-40GB | Causal reranker via /predict |
| BEI-qwen-qwen3-reranker-8b-fp8 | H100-40GB | Causal reranker via /predict |
| BEI-skywork-skywork-reward-llama-3.1-8b-v0.2-reward-model-fp8 | H100-40GB | Reward model |
| BEI-qwen-qwen3-reranker-8b-fp4 | B200 | FP4 variant — change `quantization-type` to `fp4` |

---

## `bei-bert-embedder.yaml`
`base-model: encoder-bert` · `/v1/embeddings` · no quantization

| Config folder | Accelerator |
|---|---|
| BEI-Bert-alibaba-nlp-gte-modernbert-base-embedding | L4 |
| BEI-Bert-alibaba-nlp-gte-qwen2-1.5b-instruct-embedding | L4 |
| BEI-Bert-alibaba-nlp-gte-qwen2-7b-instruct-embedding | H100 |
| BEI-Bert-google-embeddinggemma-300m | L4 |
| BEI-Bert-intfloat-multilingual-e5-large-instruct | L4 |
| BEI-Bert-jina-ai-jina-embeddings-v2-base-en | L4 |
| BEI-Bert-jinaai-jina-embeddings-v2-base-code | L4 |
| BEI-Bert-mixedbread-ai-mxbai-embed-large-v1-embedding | L4 |
| BEI-Bert-nomic-ai-nomic-embed-text-v1.5 | A10G |
| BEI-Bert-nomic-ai-nomic-embed-text-v2-moe | L4 |
| BEI-Bert-nvidia-llama-embed-nemotron-8b | H100 |
| BEI-Bert-nvidia-llama-nemotron-embed-1b-v2 | H100 |
| BEI-Bert-redis-langcache-embed-v2 | L4 |
| BEI-Bert-sentence-transformers-all-minilm-l6-v2-embedding | L4 |
| BEI-Bert-taylorai-bge-micro-v2 | A10G |
| BEI-Bert-voyageai-voyage-4-nano | L4 |

---

## `bei-bert-reranker.yaml`
`base-model: encoder-bert` · `/rerank` · no quantization

| Config folder | Accelerator | Notes |
|---|---|---|
| BEI-Bert-alibaba-nlp-gte-reranker-modernbert-base | L4 | |
| BEI-Bert-baai-bge-reranker-large | H100 | |
| BEI-Bert-ner-bert-base-ner-uncased | L4 | ⚠️ Named "NER" but generated with `/rerank` route |

---

## `bei-bert-predictor.yaml`
`base-model: encoder-bert` · `/predict` · no quantization

| Config folder | Accelerator |
|---|---|
| *(none currently generated)* | |

---

## `bei-bert-ner.yaml`
`base-model: encoder-bert` · `/predict-tokens` · no quantization

| Config folder | Accelerator |
|---|---|
| BEI-Bert-babelscape-wikineural-multilingual-ner | L4 |
| BEI-Bert-dslim-bert-base-ner-uncased | L4 |
| BEI-Bert-lcampillos-roberta-es-clinical-trials-ner | L4 |
| BEI-Bert-tanaos-tanaos-ner-v1 | L4 |

---

## `tei-embedder.yaml`
Docker-based HuggingFace TEI · weights baked in at build time

| Config folder | Accelerator | Task |
|---|---|---|
| TEI-alibaba-nlp-gte-modernbert-base-embedding | L4 | Embedding |
| TEI-alibaba-nlp-gte-qwen2-1.5b-instruct-embedding | L4 | Embedding |
| TEI-alibaba-nlp-gte-qwen2-7b-instruct-embedding | H100 | Embedding |
| TEI-alibaba-nlp-gte-reranker-modernbert-base | L4 | Reranker (change `predict-endpoint` to `/rerank`) |
| TEI-baai-bge-reranker-large | H100 | Reranker (change `predict-endpoint` to `/rerank`) |
| TEI-google-embeddinggemma-300m | L4 | Embedding |
| TEI-intfloat-multilingual-e5-large-instruct | L4 | Embedding |
| TEI-jina-ai-jina-embeddings-v2-base-en | L4 | Embedding |
| TEI-jinaai-jina-embeddings-v2-base-code | L4 | Embedding |
| TEI-mixedbread-ai-mxbai-embed-large-v1-embedding | L4 | Embedding |
| TEI-nomic-ai-nomic-embed-text-v1.5 | A10G | Embedding |
| TEI-nomic-ai-nomic-embed-text-v2-moe | L4 | Embedding |
| TEI-redis-langcache-embed-v2 | L4 | Embedding |
| TEI-sentence-transformers-all-minilm-l6-v2-embedding | T4 | Embedding |
| TEI-taylorai-bge-micro-v2 | A10G | Embedding |


> Briton and BISV2 configs are mapped in `briton-bisv2-decoder/` instead.
