# Zembed 1 on TEI

Serve [ZeroEntropy's zembed-1-embedding](https://huggingface.co/zeroentropy/zembed-1-embedding), an Apache-2.0, multilingual, 4B-parameter Qwen3-based text-embedding model, with Hugging Face Text Embeddings Inference (TEI) 1.9. This config is the registry copy of the one serving the ZeroEntropy lab listing `zeroentropy/zembed-1`: one `H100`, float16, an 8,192-token server batch budget, up to 128 inputs per request, 4,096 concurrent requests, an 8 MiB payload limit, and `document` as the default prompt. It is deploy-only and publishes no listing of its own; see the last section.

## Checkpoint

The config pins revision `cf13c81f3274394053d166740294f7eea4586f7a` (2026-07-24), the commit that relicensed the repo under Apache-2.0. The deployed listing pins the earlier `10378878bba40172305a1a979db64a413ab7da7b` (2026-03-12). Both safetensors shards, `projections.safetensors`, the tokenizer files, `modeling_zembed.py` and every config file have identical hashes at the two revisions; the later commit only adds `LICENSE` and changes the README license from CC BY-NC 4.0 to Apache-2.0, so the same weights are served. The repo is public and ungated, so no `hf_access_token` is declared.

The checkpoint's `sentence_bert_config.json` sets `max_seq_length` to 32,768, the context the model card documents, although the underlying Qwen3 config allows 40,960 positions. TEI runs the Qwen3 architecture natively and does not execute the repo's Python module.

## Prompts and dimensions

Zembed 1 is instruction-aware. Its `config_sentence_transformers.json` defines two prompts, `query` and `document`, each a chat-formatted system/user prefix with an `<|im_end|>` suffix, and TEI applies them by name. The server defaults to `document`; send `prompt_name: query` for search queries and leave it unset, or send `document`, for passages. Mixing the two conventions degrades retrieval quality.

The model card lists output sizes of 2,560 (native), 1,280, 640, 320, 160, 80 and 40 produced by learned projection layers. TEI's `dimensions` parameter truncates the output vector Matryoshka-style, which is not that projection method. The deployed playground example requests 1,280 dimensions this way; evaluate retrieval quality at reduced dimensions on your own data before relying on it, or use the full 2,560-dimensional output.

## Call the model

POST to `https://model-MODEL_ID.api.baseten.co/environments/production/sync/predict`. Baseten routes it to TEI's `/embed`. The request takes `inputs` (a string or a list of up to 128 strings), an optional `prompt_name`, and optional `dimensions`, `normalize` (default true) and `truncate` (default false). Inputs longer than 32,768 tokens are rejected unless `truncate: true`. The response is a list of float vectors, one per input.

```bash
curl -s -X POST "https://model-MODEL_ID.api.baseten.co/environments/production/sync/predict" \
  -H "Authorization: Api-Key $BASETEN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": ["What is backpropagation?"],
    "prompt_name": "query",
    "dimensions": 1280
  }'
```

For documents, omit `prompt_name` or set it to `document`. TEI also serves an OpenAI-compatible `/v1/embeddings` route, which has no `prompt_name` field.

## Relationship to the ZeroEntropy lab listing

ZeroEntropy's lab listing `zeroentropy/zembed-1` is served from this config. This preset is its registry copy: the README declares no `library_id`, so registry CI deploys and benchmarks it but does not create a separate `baseten/…` listing beside the lab-owned one. The `lab_listing` field above records which listing the config serves.