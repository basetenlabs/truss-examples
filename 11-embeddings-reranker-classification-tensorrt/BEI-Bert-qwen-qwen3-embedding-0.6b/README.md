# Qwen3 Embedding 0.6B

Deploy [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) using BEI-Bert's `encoder_bert` integration. The config pins the checkpoint revision and serving version.

## Deploy

From the repository root:

```bash
pip install --upgrade truss
truss push 11-embeddings-reranker-classification-tensorrt/BEI-Bert-qwen-qwen3-embedding-0.6b --publish
```

Prefix retrieval queries with the task instruction shown below. Pass document text without a query instruction when building the document index.

Inputs longer than the configured token budget are truncated. Choose a limit appropriate for your documents and GPU memory.

## Invoke

Set `MODEL_ID` to your deployment's model ID and `BASETEN_API_KEY` to your API key:

```bash
export MODEL_URL="https://model-${MODEL_ID}.api.baseten.co/environments/production/sync"
curl --fail-with-body "$MODEL_URL/v1/embeddings" \
  -H "Authorization: Api-Key $BASETEN_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
  "input": [
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: What is the capital of France?"
  ],
  "model": "Qwen/Qwen3-Embedding-0.6B"
}'
```

## Multiple GPUs

For independent replicas in one deployment, change `resources.accelerator` to `L4:2` or `L4:4`. Each GPU holds the whole model; weights are not split between GPUs. Start with one L4, then measure throughput and queueing at your target load before adding replicas. RTX6000 deployments follow the same full-model-per-GPU capacity requirement where that accelerator is available.
