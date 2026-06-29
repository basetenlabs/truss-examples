# Embedding Model Config Templates

These are base `config.yaml` templates for deploying models on Baseten via Truss. Copy the right template, fill in the placeholders, and run `truss push --publish`.

---

## How to pick a template

### Step 1 — What kind of model is it?

Look at the model's HuggingFace page. The architecture suffix in the model card tells you the task:

| HuggingFace architecture suffix | Task | Template(s) |
|---|---|---|
| (sentence-transformers files: `sbert_config.json` or `1_Pooling/config.json`) | Embedder | `bei_embedder*`, `bei_bert_embedder`, `tei_embedder` |
| `ForSequenceClassification` with **1 label**, BERT-family | Reranker | `bei_reranker`, `bei_bert_reranker` |
| `ForSequenceClassification` with **1 label**, causal (Llama/Qwen-based) | Reranker via `/predict` | `bei_predictor*` |
| `ForSequenceClassification` with **multiple labels** | Classifier/Predictor | `bei_predictor*`, `bei_bert_predictor`* |
| `ForTokenClassification` | NER | `bei_bert_ner` |

*`bei_bert_predictor.yaml` exists as a template but has no examples in the current model library yet.

> For text generation LLMs (ForCausalLM), see the `briton_bisv2_templates` directory instead.

---

### Step 2 — Which inference engine?

**For embedding, reranker, and classifier models**, choose between BEI and BEI-Bert:

| Condition | Engine | Notes |
|---|---|---|
| Causal architecture used as encoder (Llama, Mistral, Qwen2 variants) | **BEI** (`bei_*.yaml`) | `base_model: encoder` |
| BERT/RoBERTa/ModernBERT bidirectional architecture | **BEI-Bert** (`bei_bert_*.yaml`) | `base_model: encoder_bert` |
| Model not yet supported by BEI or BEI-Bert | **TEI** (`tei_embedder.yaml`) | Docker-based fallback; weights baked in at build time |

A quick way to tell: if the HuggingFace model card mentions `sentence-transformers` and the model has a `config.json` with `"model_type": "bert"`, `"roberta"`, or `"modernbert"` → use BEI-Bert. If `"model_type"` is `"llama"`, `"mistral"`, `"qwen2"`, or similar causal type → use BEI.

---

### Step 3 — Quantization?

| Quantization | GPU requirement | Template suffix | When to use |
|---|---|---|---|
| None | Any | (no suffix) | Small models (<1B), maximum accuracy, or pre-quantized weights (see below) |
| FP8 | H100, H100_40GB, L4 | `_fp8` | Best default for larger models |
| FP4 | B200 only | `_fp4` | Maximum throughput on B200 |

Note: BEI-Bert does **not** support FP8 or FP4.

**Pre-quantized weights:** Some HuggingFace repos ship weights that are already quantized. For these, use `quantization_type: no_quant` even on B200 — the quantization happened offline, not at build time.

---

## Fill in the placeholders

Every template has these placeholders:

| Placeholder | What to put |
|---|---|
| `<org>/<model-name>` | The HuggingFace repo ID, e.g. `BAAI/bge-large-en-v1.5` |
| `model_name: BEI-<org>-...` | A human-readable name for your deployment |
| `accelerator` | GPU type: `L4`, `A10G`, `H100`, `H100_40GB`, `H100_MIG_1G.20GB`, `B200` |
| `max_num_tokens` | Set to `max(16384, model.max_position_embeddings)`. Find `max_position_embeddings` in the model's `config.json` on HuggingFace. |
| `num_builder_gpus` | Only relevant for FP8/FP4. Set to `2` for H100, `4` for L4. |

---

## GPU sizing guide

| Model size | Recommended GPU |
|---|---|
| < 1B params | L4 or A10G |
| 1B–7B params | L4 or H100_40GB |
| 7B–13B params | H100_40GB |
| > 13B params or long context (>8K tokens) | H100 |
| FP4 quantization | B200 |

---

## Deploy

```sh
# Install Truss
pip install --upgrade truss

# Deploy (from the directory containing your config.yaml)
truss push --publish
```

If the model is gated on HuggingFace, add your token as a Baseten secret at https://app.baseten.co/settings/secrets with the key `hf_access_token`, then add to your config:

```yaml
secrets:
  hf_access_token: null
```
