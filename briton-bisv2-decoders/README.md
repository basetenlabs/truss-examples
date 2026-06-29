# Engine Builder Config Templates

These are base `config.yaml` templates for deploying LLMs on Baseten via Truss and the Engine Builder. Copy the right template, fill in the placeholders, and run `truss push --publish`.

Engine Builder compiles your model into a TensorRT-LLM engine at deploy time, optimized for the specific GPU you choose. This compilation happens once; the resulting engine runs with lower latency and higher throughput than loading weights directly.

All deployments expose an OpenAI-compatible API at `/v1/chat/completions`.

---

## Base templates

| Template | Engine | Quantization | GPU |
|---|---|---|---|
| `briton-decoder.yaml` | Briton (v1) | none (bf16/fp16) | Any |
| `briton-decoder-fp8.yaml` | Briton (v1) | fp8 / fp8-kv | H100, H200, L4 |
| `briton-decoder-fp4.yaml` | Briton (v1) | fp4 / fp4-kv / fp4-mlp-only | B200 |
| `briton-decoder-speculative-lookahead.yaml` | Briton (v1) | any (+ lookahead) | H100, H200 |
| `bisv2-decoder-fp8.yaml` | BISV2 (v2) | fp8 / fp8-kv | H100 |
| `bisv2-decoder-fp4.yaml` | BISV2 (v2) | fp4 / fp4-kv / fp4-mlp-only | B200 |
| `bisv2-decoder-prequantized.yaml` | BISV2 (v2) | no-quant (pre-quantized weights) | matches checkpoint |

See `mappings.md` for which model in `archive/11-embeddings-reranker-classification-tensorrt` maps to each template.

---

## How to pick a template

### Step 1 — Which engine?

| Condition | Template family |
|---|---|
| Dense model (Llama, Qwen, Mistral, Gemma, Falcon, Phi) | `briton-decoder*.yaml` |
| MoE model (DeepSeek, Qwen3-MoE) or multi-node setup | `bisv2-decoder*.yaml` |

**Briton** uses inference stack v1, which gives you more control over build parameters and supports speculative decoding and LoRA adapters.

**BISV2** uses inference stack v2, which manages most build parameters automatically. It's better suited for large MoE models and multi-node configurations but has fewer knobs to turn.

### Step 2 — Quantization?

| Quantization | GPU | Template | When to use |
|---|---|---|---|
| `no-quant` | Any | `briton-decoder.yaml` / `bisv2-decoder-prequantized.yaml` | Small models (<7B), maximum accuracy, or pre-quantized weights (see below) |
| `fp8-kv` | H100, H200, L4 | `briton-decoder-fp8.yaml` / `bisv2-decoder-fp8.yaml` | Best default for most models ≥7B |
| `fp8` | H100, H200, L4 | `briton-decoder-fp8.yaml` / `bisv2-decoder-fp8.yaml` | Qwen2-family models only (fp8-kv has quality issues with Qwen2); add `quantization-config: {calib-size: 2048}` |
| `fp4` / `fp4-kv` | B200 only | `briton-decoder-fp4.yaml` / `bisv2-decoder-fp4.yaml` | Maximum throughput on B200 |
| `fp4-mlp-only` | B200 only | `briton-decoder-fp4.yaml` / `bisv2-decoder-fp4.yaml` | Conservative FP4 — only quantizes MLP layers, better quality |

**Pre-quantized weights:** Some HuggingFace repos (e.g. `nvidia/Llama-3.1-8B-Instruct-FP4`) ship weights that are already quantized. For these, use `bisv2-decoder-prequantized.yaml` with `quantization-type: no-quant` — the quantization happened offline, not at build time.

### Step 3 — Tensor parallelism?

Set `tensor-parallel-count` equal to the number of GPUs you want to use (Briton only; BISV2 manages this automatically). The `resources.accelerator` field takes a count suffix, e.g. `H100:2` for 2× H100.

For single-GPU Briton builds with quantization, set `num-builder-gpus: 4` to speed up compilation. Omit it when `tensor-parallel-count > 1`.

---

## Speculative decoding

Use `briton-decoder-speculative-lookahead.yaml` for **lookahead decoding** — self-speculative, no draft model needed. It layers a `speculator` block on top of `briton-decoder-fp8.yaml` and works with any `quantization-type` (no-quant, fp8, fp8-kv).

Briton also supports a **draft-model** mode: add a `speculator` block with `speculative-decoding-mode: DRAFT-TOKENS-EXTERNAL` and a `checkpoint-repository` pointing to a smaller draft model. Draft and target models must share the same vocabulary size. See the `Briton-*-speculative*` examples in `archive/11-embeddings-reranker-classification-tensorrt` for the full field set.

---

## Fill in the placeholders

| Placeholder | What to put |
|---|---|
| `<org>/<model-name>` | The HuggingFace repo ID, e.g. `meta-llama/Llama-3.3-70B-Instruct` |
| `model-name` | A human-readable name for your deployment |
| `accelerator` | GPU type: `L4`, `A10G`, `H100`, `H100-40GB`, `B200`. Append `:<n>` for multi-GPU, e.g. `H100:2` |
| `max-seq-len` | Set to `model.max-position-embeddings` from the model's `config.json` on HuggingFace (BISV2 caps at 32768) |
| `tensor-parallel-count` | Briton only: number of GPUs (must match the count in `accelerator`) |

---

## GPU sizing guide

| Model size | Recommended GPU |
|---|---|
| < 7B params | H100-40GB or L4 |
| 7B–13B params | H100-40GB |
| 13B–70B params | H100 (1–2×) |
| 70B+ params | H100 (2–8×) |
| MoE models | B200 or H100 (multi-node via BISV2) |

---

## Deploy

```sh
# Install Truss
pip install --upgrade truss

# Deploy (from the directory containing your config.yaml)
truss push --publish
```

If the model is gated on HuggingFace, add your token as a Baseten secret at https://app.baseten.co/settings/secrets with the key `hf-access-token`, then add to your config:

```yaml
secrets:
  hf-access-token: null
```
