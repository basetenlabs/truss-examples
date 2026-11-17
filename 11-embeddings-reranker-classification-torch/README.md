# BEI-torch templates

Reference truss configs for the BEI-torch backend (`base_model: encoder_torch`) — the torch/vLLM
runtime used for embedding, reranker, and classification models whose architecture the
TRT-LLM `encoder`/`encoder_bert` paths cannot compile (LlamaBidirectional / Ministral3-Embed,
Gemma2Embedding, jina-v3, etc.).

Templates in this directory are hand-written pending integration with
`../11-embeddings-reranker-classification-tensorrt/templating/generate_templates.py`.
When picking `encoder_torch` vs `encoder`/`encoder_bert`, use:

- **`encoder_bert`** — BERT-family (BERT, ModernBERT, XLM-R, jina-v2). Fastest cold-start.
- **`encoder`** — causal-arch models with FP8 support on H100/B200 (Qwen3-Embedding, SFR-Embedding).
- **`encoder_torch`** — everything else. LlamaBidirectional (Nemotron-3-Embed), Gemma2Embedding,
  jina-v3, custom architectures BEI-torch supports but TRT-LLM does not.

## Known issue with stock NVIDIA Nemotron-3-Embed repos

The stock `nvidia/Nemotron-3-Embed-*` repos ship JSON that HF `transformers` rejects on load:

1. `tokenizer_config.json` has `tokenizer_class: "TokenizersBackend"` (not a real class).
2. `tokenizer_config.json` has `extra_special_tokens` as a list; transformers expects a dict.
3. `config.json` has `model_type: "llama_bidirec"` which is not registered with transformers.

Until the BEI-torch backend lands the `llama_bidirec` serde alias and a tokenizer-config
fallback, point `checkpoint_repository.repo` at a fork with those three fields patched.
See the per-template README for the exact patch set.
