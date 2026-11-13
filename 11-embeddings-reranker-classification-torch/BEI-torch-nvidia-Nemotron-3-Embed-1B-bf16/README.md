# BEI-torch with nvidia/Nemotron-3-Embed-1B-BF16

Deployment for [Nemotron-3-Embed-1B-BF16](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16)
on the BEI-torch backend. Native embedding dim 2048; MRL heads at 512 / 1024 / 2048.

Fits on an L4 (24 GB) at bf16.

## Prerequisites

1. [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Latest truss: `pip install --upgrade truss`
3. Set `hf_access_token` as a [Baseten secret](https://app.baseten.co/settings/secrets).

## Known issue: stock `nvidia/Nemotron-3-Embed-1B-BF16` fails to load

Same three JSON patches apply as for the 8B model — see
[`../BEI-torch-nvidia-Nemotron-3-Embed-8B-bf16/README.md`](../BEI-torch-nvidia-Nemotron-3-Embed-8B-bf16/README.md).
Point `checkpoint_repository.repo` at a fork with those patches applied.

## Deploy

```sh
git clone https://github.com/basetenlabs/truss-examples.git
cd 11-embeddings-reranker-classification-torch/BEI-torch-nvidia-Nemotron-3-Embed-1B-bf16
truss push --publish
```

## Call

```bash
curl -X POST https://model-xxxxxx.api.baseten.co/environments/production/sync/v1/embeddings \
     -H "Authorization: Api-Key $BASETEN_API_KEY" \
     -d '{"input": "text string", "model": "model"}'
```
