# NVIDIA Nemotron 3 Nano 30B A3B — latency preset

NVIDIA Nemotron 3 Nano 30B A3B is a **text-only** open reasoning model with a
**Mamba2-Transformer hybrid MoE** architecture: 52 layers — 23 Mamba-2, 23 MoE and 6
grouped-query-attention — with 128 routed experts plus 1 shared expert per MoE layer and 6
activated per token. That gives **3.5B active parameters out of 30B total**, which is what
makes a 30B-class model viable on a single GPU at voice latencies.

Served via **vLLM** (OpenAI-compatible `/v1/chat/completions`) on a **single H100** from the
**FP8** checkpoint — weights *and* KV cache are FP8, with attention and the Mamba layers
feeding it kept in BF16.

Not to be confused with [`nemotron-3-nano-omni`](../../nemotron-3-nano-omni/latency/), which
is the multimodal *omni* variant. This entry is the text-only 30B-A3B, the LLM stage of the
NVIDIA cascaded voice pipeline.

Listed in the **Model Library** as *NVIDIA Nemotron 3 Nano 30B A3B*. `public: true` also lets
users download this truss config and pushes it to `truss-examples` on merge.

Note this is the **latency** preset: `--max-num-seqs 8` and a 32k window are sized for the
voice pipeline, not for batch throughput. A throughput preset would raise both.

## Model details

- **Source weights:** `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` @
  `f8dc1c0afee92f44417695b4f5ddca9afc95ea58` — public and ungated, pulled with the Baseten
  secret `hf_access_token`.
- **Served model name:** `nvidia/nemotron-3-nano`.
- **Context:** 32768. The model card defaults to 262144 and supports up to 1M with
  `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`; 32k matches NVIDIA's voice blueprint and leaves KV
  headroom on one H100. Raise it (and lower `--max-num-seqs`) for long-context use.
- **Concurrency:** `--max-num-seqs 8`, `predict_concurrency: 8`. Higher concurrency belongs
  in a `throughput` preset.

## Reasoning

### The parser is `nano_v3`, not `nemotron_v3`

This model ships **its own reasoning parser inside the HF repo**
(`nano_v3_reasoning_parser.py`), loaded as a plugin:

```
--reasoning-parser-plugin /app/checkpoint/model/nano_v3_reasoning_parser.py
--reasoning-parser nano_v3
```

`nemotron_v3` is the built-in parser used by the **omni** entry and by the 2.x LLM **NIM** —
it is *not* the parser for this model. Don't copy it across. Without a working parser the
`<think>…</think>` trace lands in `content`, which in a voice agent means TTS reads the
chain-of-thought aloud.

The model card fetches the plugin with `wget`; that's unnecessary here, since `weights:`
mounts the whole repo at `/app/checkpoint/model`.

### Controlling it

Reasoning is enabled by default. Turn it off per request:

```json
{"chat_template_kwargs": {"enable_thinking": false}}
```

`enable_thinking` is the key the parser reads; `thinking` is not used.

With reasoning on, budget a **high `max_tokens`** — the model card suggests ~10,000. The
trace is emitted *before* the answer, so too small a budget returns `finish_reason="length"`
and no usable content at all.

Note that "on by default" does not mean a trace is guaranteed: the parser splits on `<think>`
tags, so `reasoning_content` is populated only when the model actually emits a block, which
is prompt- and sampling-dependent. Treat its presence as optional in client code.

### Known gap: no reasoning-token count

`usage.completion_tokens_details.reasoning_tokens` is **absent** — the key is not emitted at
all. This is upstream, not a config knob: NVIDIA's parser subclasses vLLM's
`DeepSeekR1ReasoningParser` and overrides only `extract_reasoning`, doing no token
accounting. The reasoning *content* splits correctly; only the *count* is unavailable, so
anything billing or budgeting on `reasoning_tokens` must derive it itself.

## Recommended generation settings

- Non-thinking: `temperature=0.2`
- Thinking: `temperature=0.6`, `top_p=0.95`, and a large `max_tokens`

## Example request

Reasoning off, streaming — the voice-pipeline shape:

```python
from openai import OpenAI

client = OpenAI(base_url="https://<your-deployment>/v1", api_key="<BASETEN_API_KEY>")

resp = client.chat.completions.create(
    model="nvidia/nemotron-3-nano",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    stream=True,
    max_tokens=512,
    temperature=0.2,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
for chunk in resp:
    print(chunk.choices[0].delta.content or "", end="")
```

Reasoning on — note the much larger budget:

```python
resp = client.chat.completions.create(
    model="nvidia/nemotron-3-nano",
    messages=[{"role": "user", "content": "Write a haiku about GPUs"}],
    max_tokens=10000,
    temperature=0.6,
    top_p=0.95,
)
print(resp.choices[0].message.reasoning_content)  # the trace, when emitted
print(resp.choices[0].message.content)            # the answer
```

## Tool calling

Enabled via `--enable-auto-tool-choice --tool-call-parser qwen3_coder`; pass `tools` as
usual. `qwen3_coder` is what NVIDIA specifies for this model.

## Serving flags

| Flag / env | Why |
|---|---|
| `--kv-cache-dtype fp8` | The checkpoint quantized the KV cache to FP8 along with the weights. |
| `VLLM_USE_FLASHINFER_MOE_FP8=1` | FlashInfer FP8 MoE kernels, per the model card. |
| `--mamba-ssm-cache-dtype float32` | Keeps the hybrid Mamba SSM state cache in fp32 to avoid accuracy drift. Beyond the card; matches the omni entry. |
| `--trust-remote-code` | The repo ships `modeling_nemotron_h.py` / `configuration_nemotron_h.py`. |
| `--load-format runai_streamer` | Faster cold start on ~32GB of shards. |
| `--tensor-parallel-size 1` | Single-GPU preset, so it needs no `sh -c` wrapper at all. Multi-GPU entries derive TP from `nvidia-smi` and must single-quote the wrapper (#283) to defer expansion into the container; invoking vLLM directly avoids that class of quoting bug entirely. |

### Weights come from the upstream repo, not a mirror

Called out because a NIM-based deployment of this model exists and the reason is easy to
misremember. An earlier attempt failed with an HF **Xet** range-request `403` on every large
shard, but that was under **`model_cache:`** and is specific to that path — not a property of
this repo, and not something a mirror would fix (`baseten-admin` is itself Xet-migrated).
Pulling the same repo through `weights:` works: 33 GB in 34s on first deploy.

So this entry needs no mirror, no NGC entitlement and no rehosted image.

## Benchmarking

Picked up automatically by the `llm` modality spec
([`run-b10-bench/modalities/llm/spec.py`](../../../.github/actions/run-b10-bench/modalities/llm/spec.py))
— a `perf.aiperf` (ISL, OSL) × concurrency sweep. Dispatch `model_benchmark.yml` with
`model-directory: llm/nemotron-3-nano/latency`. The `full` profile's largest scenario is
`D(32768, 1024)`, which exceeds the served context and is gated out.

### Measured — `smoke` profile, 1×H100

From the pipeline above (run `run-9fe628036013`), 0.00% errors on all four sub-runs.

| Scenario | Conc | TTFT p50 | TTFT p99 | TPOT p50 | Output tok/s | E2E p50 |
|---|---|---|---|---|---|---|
| D(128,128) | 1 | 178 ms | 268 ms | 3.79 ms | 187 | 661 ms |
| D(128,128) | 4 | 249 ms | 328 ms | 5.87 ms | 500 | 999 ms |
| D(1024,1024) | 1 | 181 ms | 1277 ms | 4.24 ms | 223 | 4522 ms |
| D(1024,1024) | 4 | 189 ms | 380 ms | 6.67 ms | 640 | 7052 ms |

**Treat the concurrency-4 rows as indicative, not a floor.** Two CI runs of this same
config measured TPOT p50 at 3.5 ms and 5.9 ms for D(128,128) c=4, with output throughput
751 vs 500 tok/s — roughly a 1.5–2x spread that is run-to-run, not configuration. TTFT and
the concurrency-1 rows are stable across runs. If you need a number to plan against, use
the slower one here.

For reference, the NIM deployment this replaces measured TTFT p50 0.231 s warm against
0.178 s here — the vLLM path costs nothing in latency.