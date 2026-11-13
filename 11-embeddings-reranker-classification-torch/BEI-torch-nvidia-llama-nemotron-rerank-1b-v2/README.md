# BEI-torch with nvidia/llama-nemotron-rerank-1b-v2

Deployment for [llama-nemotron-rerank-1b-v2](https://huggingface.co/nvidia/llama-nemotron-rerank-1b-v2)
on the BEI-torch backend. Reranker analog to Nemotron-3-Embed; scores a `(query, documents[])`
pair via `/rerank`.

Fits on an L4 (24 GB) at bf16.

## Prerequisites

1. [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Latest truss: `pip install --upgrade truss`
3. Set `hf_access_token` as a [Baseten secret](https://app.baseten.co/settings/secrets).

## Known issue: stock `nvidia/llama-nemotron-rerank-1b-v2` fails to load

Same three JSON patches apply as for the Nemotron-3-Embed models — see
[`../BEI-torch-nvidia-Nemotron-3-Embed-8B-bf16/README.md`](../BEI-torch-nvidia-Nemotron-3-Embed-8B-bf16/README.md).
Point `checkpoint_repository.repo` at a fork with those patches applied.

## Deploy

```sh
git clone https://github.com/basetenlabs/truss-examples.git
cd 11-embeddings-reranker-classification-torch/BEI-torch-nvidia-llama-nemotron-rerank-1b-v2
truss push --publish
```

## Call

```bash
curl -X POST https://model-xxxxxx.api.baseten.co/environments/production/sync/rerank \
     -H "Authorization: Api-Key $BASETEN_API_KEY" \
     -d '{
       "query": "what is baseten?",
       "documents": ["Baseten is a model deployment platform.", "Bananas are yellow."],
       "model": "model"
     }'
```
