# BEI-torch with nvidia/Nemotron-3-Embed-8B-BF16

Deployment for [Nemotron-3-Embed-8B-BF16](https://huggingface.co/nvidia/Nemotron-3-Embed-8B-BF16)
on the BEI-torch backend (text-embeddings-router + vLLM). Native embedding dim 4096; MRL heads
at 512 / 1024 / 2048 / 4096.

## Prerequisites

1. [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Latest truss: `pip install --upgrade truss`
3. Set `hf_access_token` as a [Baseten secret](https://app.baseten.co/settings/secrets).

## Known issue: stock `nvidia/Nemotron-3-Embed-8B-BF16` fails to load

Three JSON files in the stock repo trip HF `transformers`; fork the repo and apply these
patches, then set `checkpoint_repository.repo` to your fork:

1. In `tokenizer_config.json`: remove the `tokenizer_class: "TokenizersBackend"` key.
2. In `tokenizer_config.json`: remove `extra_special_tokens` (or convert the list to a dict).
3. In `config.json`: change `model_type: "llama_bidirec"` → `"llama"` and set
   `architectures: ["LlamaBidirectionalModel"]`.

The BEI-torch arch registry maps `LlamaBidirectionalModel` to a bidirectional-attention
Llama variant; `model_type: "llama"` only exists so the transformers config parser accepts
the file. This workaround will be unnecessary once the BEI-torch `llama_bidirec` serde
alias and tokenizer-config fallback land upstream.

## Deploy

```sh
git clone https://github.com/basetenlabs/truss-examples.git
cd 11-embeddings-reranker-classification-torch/BEI-torch-nvidia-Nemotron-3-Embed-8B-bf16
truss push --publish
```

## Call

```bash
curl -X POST https://model-xxxxxx.api.baseten.co/environments/production/sync/v1/embeddings \
     -H "Authorization: Api-Key $BASETEN_API_KEY" \
     -d '{"input": "text string", "model": "model"}'
```

Matryoshka embeddings (native 4096; also supports 512, 1024, 2048):

```bash
curl -X POST https://model-xxxxxx.api.baseten.co/environments/production/sync/v1/embeddings \
     -H "Authorization: Api-Key $BASETEN_API_KEY" \
     -d '{"input": "text string", "model": "model", "dimensions": 1024}'
```
