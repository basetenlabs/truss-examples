# Pipecat PhoneLLM Alpha 1 — latency preset

Pipecat PhoneLLM Alpha 1 is a **text-in / text-out** open-weights LLM tuned for voice-agent
phone conversation and tool calling. It has **no audio path at all** — no encoder, no
decoder, no `preprocessor_config.json`. It is the **LLM stage of a cascaded voice
pipeline**: Pipecat pairs it with separate transcription and text-to-speech models to
answer calls. Do not look for STT or TTS behaviour here, and do not file it under `stt/`
or `tts/`.

It is a **full-parameter SFT of
[`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)**,
so it inherits that model's **Mamba2-Transformer hybrid MoE** architecture: 52 layers —
23 Mamba-2, 23 MoE and 6 grouped-query-attention — with 128 routed experts plus 1 shared
expert and 6 activated per token. **3.5B active parameters out of 30B total.**

Served via **vLLM** (OpenAI-compatible `/v1/chat/completions`) on **one H100 (80 GiB)** from
the **BF16** checkpoint. Compare [`nemotron-3-nano`](../../nemotron-3-nano/latency/), the base
model, which serves the **FP8** checkpoint — roughly half the weight footprint — on the same
single card.

Listed in the **Model Library** as *Pipecat PhoneLLM Alpha 1*. `public: true` also lets
users download this truss config and pushes it to `truss-examples` on merge.

This is the **latency** preset: the 32k window is sized for a phone call, not for batch
throughput. The in-flight cap is a separate matter — `--max-num-seqs 64` is **not** a
latency-preset default but a value sized from the checkpoint's own memory arithmetic against
the benchmark sweep's 9216-token worst case; see **Concurrency cap** below.

## Model details

- **Source weights:** `pipecat-ai/phonellm-alpha-1` @
  `8e76aaa6e8ce4765ac943ba3fb339494d4d48dca` — public and ungated, pulled with the Baseten
  secret `hf_access_token` by registry convention. 13 clean safetensors shards,
  63.16 GB / 31.58B parameters, **no `quantization_config`** — plain BF16.
- **Served model name:** `pipecat-ai/phonellm-alpha-1` — deliberately the real HF repo id
  rather than a short alias, so b10-bench's tokenizer ladder resolves it directly and this
  entry needs no `bench: {tokenizer: ...}` frontmatter override.
- **Context:** 32768. The card advertises 262144. A phone call does not reach 32k, long
  prefills are the enemy of TTFT, and 262k would cost ~1.5 GiB of KV cache *per request*
  (262144 × 6144 B). Raise it only alongside a lower `--max-num-seqs` — on one 80 GiB card
  there is no spare headroom to do both.
- **Concurrency:** `--max-num-seqs 64`, `predict_concurrency: 64`. See below.
- **License:** BSD-2-Clause, derived from NVIDIA Nemotron work. Redistribution must keep
  `LICENSE_NVIDIA.txt` and the NVIDIA notices that ship in the upstream repo.

## Why one H100

The base model's latency entry runs on one H100 because it serves an **FP8** checkpoint.
There is no FP8 release of PhoneLLM — it is BF16 only, so the weights are roughly twice the
size, 58.8 GiB, and a single 80 GiB card was the open question rather than the obvious answer.
It was benchmarked against both alternatives, and it won.

| | VRAM | Budget at `0.9` | − 58.8 GiB weights | − ~5 GiB/GPU overhead | $/hr |
|---|---:|---:|---:|---:|---:|
| **`H100` ×1 (this preset)** | 80 GiB | 72.0 GiB | 13.2 GiB | **~8.2 GiB** | **$6.50** |
| `RTX_PRO_6000` ×1 | 96 GiB | 86.4 GiB | 27.6 GiB | ~22.6 GiB | $4.00 |
| `H100:2`, TP=2 | 160 GiB | 144.0 GiB | 85.2 GiB | ~75 GiB/GPU | $13.00 |

What has to fit in ~8.2 GiB, sized against the benchmark sweep's **worst-case 9216 tokens per
request** (`(8192,1024)` and `(1024,8192)`, `benchmarking/llm/README.md:19`) rather than the
32768 ceiling:

- **KV cache — cheap.** Only 6 of this model's 52 layers are attention
  (`hybrid_override_pattern` is 23 `M` / 23 `E` / 6 `*`), with `num_key_value_heads` 2 and
  `head_dim` 128, so KV is 2(K,V) × 2 × 128 × 2 B × 6 = **6144 B/token** → 54.0 MiB/request.
- **Mamba-2 state — the expensive part, and length-independent.** 23 layers ×
  (64 × 64 × 128 × 4 B fp32 SSM = 2.00 MiB, plus 0.036 MiB conv) = **46.8 MiB/request** that
  no amount of shortening the sequence reclaims.
- **100.8 MiB per request**, plus activations, MoE all-to-all buffers, CUDA-graph capture and
  the Run:AI streamer landing buffer.

**The binding term on this card is the Mamba state, not the KV cache** — 46% of the
per-request footprint at 9216 tokens, and the majority of it below ~7600 tokens.

It fits, and it fits with room to spare at the chosen cap: the two-card default the entry
started from was more conservative than it needed to be. No setting was trimmed to make it
fit — same image, same weights pin, same `--max-model-len 32768`, same
`--gpu-memory-utilization 0.9`, same parsers and flags as the two-card and RTX PRO 6000
configurations that were benchmarked alongside it. The only differences were the accelerator,
the absent `--tensor-parallel-size`, and the in-flight cap, which is sized per card.

VRAM and price figures are from
[Baseten's instance-type table](https://docs.baseten.co/deployment/resources) (H100: 80 GiB,
$0.10833/min; RTX PRO 6000: 96 GiB, $0.06667/min).

## Concurrency cap

`--max-num-seqs 64`, `predict_concurrency: 64`. Sized from the checkpoint's own memory
arithmetic per Lei Pan's 2026-08-28 direction to cap in-flight requests by the
configuration's VRAM rather than by a flat 8 — the previous flat 8 would have made the top
four levels of the benchmark sweep (`1,4,8,16,32,64,128`) measure queueing behind an
admission gate instead of the hardware.

At 100.8 MiB per request (see above) against the ~8.2 GiB effective budget:

| Sweep level | Cache needed | On this card |
|---|---:|---|
| 1 … 32 | ≤ 3.15 GiB | comfortable |
| **64** | **6.30 GiB** | **fits, ~1.9 GiB spare — the chosen cap** |
| 128 | 12.60 GiB | **does not fit** |

The arithmetic ceiling is ~83 requests (8.2 GiB ÷ 100.8 MiB); 64 is the largest sweep level
below it. 128 needs 12.60 GiB and fails against both the conservative ~8.2 GiB budget and an
optimistic ~10.2 GiB one — **an 80 GiB card simply cannot hold 128 × 9216 tokens alongside
58.8 GiB of BF16 weights.** That is a real capacity finding about the card, not a defect in
the preset: the 96 GiB RTX PRO 6000 and the two-card TP=2 configuration both served 128
outright when they were benchmarked.

**Concurrency 128 is out of reach on 80 GiB.** Any c=128 cell measured against this config is
an admission-queue measurement, not a latency measurement, and must be reported that way.

## Reasoning and sampling

**Upstream is explicit: set `temperature` to 0 and disable thinking.** Both are per-request
settings, not server flags, so they live in `model_metadata.example_model_input` and in the
snippets below — the server cannot enforce them.

Disable thinking with:

```json
{"chat_template_kwargs": {"enable_thinking": false}}
```

The reasoning parser is still loaded server-side:

```
--reasoning-parser-plugin /app/checkpoint/model/nano_v3_reasoning_parser.py
--reasoning-parser nano_v3
```

`nano_v3_reasoning_parser.py` ships **inside this HF repo** (it imports `vllm.reasoning.*`,
so it is vLLM-only), and `weights:` mounts the whole repo, so the plugin is already on disk.
This is **not** the built-in `nemotron_v3` parser. Keeping it loaded means that if a caller
does turn thinking on, the trace lands in `reasoning_content` instead of in `content` —
which in a voice agent is the difference between a hidden trace and TTS reading the model's
chain-of-thought down the phone line.

Two upstream inconsistencies worth knowing, neither of which needs a config change:

- `generation_config.json` sets `do_sample: true`, which contradicts the card's
  `temperature: 0`. vLLM's default `--generation-config auto` reads that file, so **the
  request must set `temperature: 0` explicitly**.
- `eos_token_id` is `2` in `config.json` but `[2, 11]` in `generation_config.json`. Leaving
  `--generation-config` at `auto` is deliberate: it picks up the two-element list, which is
  the correct stop set.

## Example request

The voice-pipeline shape — thinking off, temperature 0, streaming:

```python
from openai import OpenAI

client = OpenAI(base_url="https://<your-deployment>/v1", api_key="<BASETEN_API_KEY>")

resp = client.chat.completions.create(
    model="pipecat-ai/phonellm-alpha-1",
    messages=[
        {"role": "system", "content": "You are a helpful voice assistant answering a phone call. Keep replies short and speakable."},
        {"role": "user", "content": "Hi, I'd like to move my appointment to next Tuesday."},
    ],
    stream=True,
    max_tokens=512,
    temperature=0,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
for chunk in resp:
    print(chunk.choices[0].delta.content or "", end="")
```

## Tool calling

Enabled with `--enable-auto-tool-choice --tool-call-parser qwen3_coder`; pass `tools` as
usual. `qwen3_coder` is what upstream specifies for this model family — it is inherited
from the Nemotron 3 Nano base recipe, not a guess.

## Serving flags

| Flag / env | Why |
|---|---|
| *(no `--tensor-parallel-size`)* | One card. `resources.accelerator` pins the count, so nothing is derived from `nvidia-smi` — which means no `sh -c` wrapper and none of the quote-stripping failures described in [`llm/AGENT.md`](../../AGENT.md). |
| `--trust-remote-code` | The repo ships `configuration_nemotron_h.py` / `modeling_nemotron_h.py` and `config.json` has an `auto_map`. vLLM has a native `NemotronHForCausalLM` implementation so the remote modeling code should be bypassed at serve time, but both upstream recipes pass this flag and the card asks for it. |
| `--mamba-ssm-cache-dtype float32` | Matches the checkpoint's own `mamba_ssm_cache_dtype`; keeps the hybrid Mamba SSM state in fp32 to avoid accuracy drift. |
| `--max-model-len 32768` | Voice-pipeline sizing, not the card's 262144. See Model details. |
| `--max-num-seqs 64` | Sized to this card's VRAM. See Concurrency cap. |
| `--enable-prefix-caching` | System prompts are re-sent on every turn of a call; caching the prefix is close to free TTFT. |
| `--load-format runai_streamer` | Faster cold start over 63 GB of shards. |
| `--gpu-memory-utilization 0.9` | Repo default for this family. |

### Deliberately not carried over from the FP8 base entry

| Not set | Why |
|---|---|
| `--kv-cache-dtype fp8` | The base entry quantizes the KV cache to FP8 because *its checkpoint* is FP8. This one is plain BF16 with no `quantization_config`; forcing FP8 KV would change numerics upstream never validated. |
| `VLLM_USE_FLASHINFER_MOE_FP8=1` | FlashInfer **FP8** MoE kernels, for an FP8 checkpoint. Irrelevant to a BF16 one. |

## Benchmarking

Picked up automatically by the `llm` modality spec
([`run-b10-bench/modalities/llm/spec.py`](../../../.github/actions/run-b10-bench/modalities/llm/spec.py))
— a `perf.aiperf` (ISL, OSL) × concurrency sweep. No frontmatter opt-in is required.
Dispatch `model_benchmark.yml` with `model-directory: llm/phonellm-alpha-1/latency`.

Note on scenario gating: a scenario is dropped when `isl + osl > max_context`. Of the six
`full` scenarios only `D(32768, 1024) = 33792` exceeds the served 32768 —
`D(8192, 8192) = 16384` is comfortably inside it and does run. Five of six survive.

### Measured — how this configuration was chosen

Six configurations — {vLLM, SGLang} × {2× H100, 1× H100, RTX PRO 6000} — were run on
`profile:full` across `D(128,128)`, `D(1024,1024)` and `D(8192,1024)` at concurrencies
1, 4, 8, 16, 32, 64, 128. **Only the winner is kept in the registry**; the other five were
benchmarked and then removed. The per-cell tables are the store of record:

- [vLLM / 1× H100 / cap 64 — all three scenarios (this configuration)](https://github.com/basetenlabs/model-registry/pull/353#issuecomment-5497449359)
- [vLLM / H100:2 / cap 128 — D(128,128) + D(1024,1024)](https://github.com/basetenlabs/model-registry/pull/353#issuecomment-5484661435)
- [vLLM / H100:2 / cap 128 — D(8192,1024)](https://github.com/basetenlabs/model-registry/pull/353#issuecomment-5497448927)
- [vLLM / RTX PRO 6000 / cap 128 — all three scenarios](https://github.com/basetenlabs/model-registry/pull/353#issuecomment-5497449134)
- SGLang arms, in [#356](https://github.com/basetenlabs/model-registry/pull/356):
  [H100:2](https://github.com/basetenlabs/model-registry/pull/356#issuecomment-5497449559),
  [RTX PRO 6000](https://github.com/basetenlabs/model-registry/pull/356#issuecomment-5497449717),
  [1× H100](https://github.com/basetenlabs/model-registry/pull/356#issuecomment-5497449878)

At `D(1024,1024)`. Latency columns are concurrency 1; peak output tok/s is the best cell
anywhere in that arm's sweep for this scenario, and the cost column is that peak cell's. Cost
is derived, not measured — `$/M out tok = (hourly rate / 3600) / output_tok_s × 1e6`, instance
rent over measured throughput — so it excludes cold starts, idle time and replica headroom and
should be read as a floor.

| Configuration | $/hr | p50 TTFT @ c=1 | p50 TPOT @ c=1 | Peak output tok/s | $/M output tok |
|---|---:|---:|---:|---:|---:|
| vLLM, 2× H100 | $13.00 | 100.33 ms | 3.52 ms | 7140.71 (c=128) | **$0.51** |
| **vLLM, 1× H100 (this preset)** | **$6.50** | **139.07 ms** | **4.99 ms** | **2724.63 (c=64)** | **$0.66** |
| SGLang, 1× H100 | $6.50 | 214.79 ms | 5.01 ms | 1439.32 (c=128) | $1.25 |
| vLLM, RTX PRO 6000 | $4.00 | 154.07 ms | 11.13 ms | 883.86 (c=64) | $1.26 |
| SGLang, 2× H100 | $13.00 | 133.86 ms | 4.54 ms | 2840.26 (c=128) | $1.27 |
| SGLang, RTX PRO 6000 | $4.00 | 134.70 ms | 11.09 ms | 591.41 (c=64) | $1.88 |

**139 ms to first token is inside a conversational turn budget**, and ~5 ms per output token
is roughly 200 tok/s per stream — far outrunning speech, which consumes a few tokens per
second. A phone turn is paced by TTFT and the TTS chunk size, not by this model.

Three findings from the sweep that this configuration rests on:

- **vLLM beats SGLang on every card** — ~1.5× the throughput on the RTX PRO 6000, ~1.9× on one
  H100, ~2.5× on two. That gap is far outside the run-to-run noise floor.
- **The RTX PRO 6000 has the cheapest hourly floor and the worst ceiling.** It saturates at
  884 output tok/s, and past saturation first-token time climbs into seconds — 1157 ms p50
  TTFT at c=64 and 72.7 s at c=128 on `D(1024,1024)`. Its per-token decode is also ~2.2×
  slower than an H100's; vLLM ships no tuned Mamba `selective_state_update` config for that
  card and logs a warning saying so.
- **Two H100s are cheaper per token** ($0.51/M vs $0.66/M) **and faster to first token**
  (100 ms vs 139 ms) — but only pay off once sustained load approaches this card's ~2700
  output tok/s saturation point. Below that, the second card is idle capacity being rented.

**When to move off this configuration:** sustained load approaching **2700 output tok/s**.
That is where one card runs out; past it, switch to two H100s with `--tensor-parallel-size 2`
and a cap of 128 (`n_groups: 8` divides by 2, as do `num_key_value_heads` 2,
`mamba_num_heads` 64, `n_routed_experts` 128 and `num_attention_heads` 32). Do **not** reach
for the RTX PRO 6000, and do **not** reach for SGLang on this model.

Nothing in this section is estimated or carried over from the base model — that is a different
checkpoint at a different precision, so its numbers do not transfer. b10-bench reported **no
published baseline** for this model.

Two sampling limits to keep in mind when quoting single-stream figures: every concurrency-1
cell on a 1024-output scenario finished under 200 requests against the 600 s per-run time cap
(this configuration's c=1 cell rests on 104 completions), and 7 of the sweep's 126 cells carry
request errors of 0.20–0.78%. Prices are list rates read on 2026-08-31 and are point-in-time.

The first request after a cold start additionally JIT-compiles seven Mamba-2 Triton kernels
(`_causal_conv1d_fwd_kernel`, `_chunk_state_fwd_kernel` and friends), each logged as
`Triton kernel JIT compilation during inference ... consider extending warmup`. Extending
warmup is a reasonable follow-up if first-call latency matters.

### Why vLLM and not SGLang

The sweep settles it empirically — vLLM wins by 1.5–2.5× on every card — but the reasoning
that led to trying vLLM first is worth recording, and so is a correction.

Both engines support the architecture natively (`NemotronHForCausalLM` is in vLLM's
`registry.py` and in SGLang's `EntryClass`), and the card says Pipecat uses both. The
serving-stack ladder is top-down and vLLM sits above SGLang, so vLLM wins unless SGLang has
a concrete advantage.

**Correction.** An earlier revision of this section justified the choice by claiming SGLang
"names no stable release" and that an SGLang variant would require pinning a nightly, against
repo convention. **That was false.** `lmsysorg/sglang:v0.5.18-cu129` is a published, versioned,
non-nightly image; its `python/sglang/srt/models/nemotron_h.py` has
`EntryClass = [NemotronHForCausalLM, NemotronHPuzzleForCausalLM]` (line 1225) and a
`NemotronHMoE` class reading `n_routed_experts` and `n_shared_experts` — the 128 routed + 1
shared shape this checkpoint has. The earliest tag carrying the class is v0.5.4; the earliest
carrying the MoE path is v0.5.6. The nightly pin in SGLang's cookbook is a **stale doc**, not
evidence that releases lack support.

SGLang was never blocked — it was **built, deployed and measured**, in
[#356](https://github.com/basetenlabs/model-registry/pull/356), and it lost. Note that the
comparison is not perfectly like-for-like: the reasoning parser differs (`nemotron_3` vs the
`nano_v3` plugin), the SGLang arms have no Run:AI streamer, SGLang has prefix caching on by
default, `--mem-fraction-static` and `--gpu-memory-utilization` do not mean the same thing,
and `--mamba-ssm-dtype float32` must be set explicitly on SGLang. No plausible share of those
five divergences closes a 2.5× throughput gap.

## Caveat: these caps assume a 9216-token worst case

`--max-num-seqs 64` was sized against the longest scenario in the benchmark sweep's `full`
profile — 9216 tokens per request (`(8192,1024)` and `(1024,8192)`,
`benchmarking/llm/README.md:19`) — **not** against the 32768-token `--max-model-len` ceiling.
If a longer scenario is ever added to the sweep, this cap goes stale and must be recomputed.
At the full 32768 tokens the per-request cost rises from 100.8 MiB to 238.8 MiB
(192.0 MiB KV + the same 46.8 MiB Mamba state), which cuts the ceiling by ~2.4× — on a single
80 GiB H100, from ~83 requests to **~35**, at which point a cap of 64 would be wrong. Anyone
changing `isl_osl_pairs`, or serving genuinely long contexts, should redo this arithmetic
before trusting the cap.