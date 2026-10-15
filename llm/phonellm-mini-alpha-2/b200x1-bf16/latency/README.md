# PhoneLLM Mini Alpha 2 — latency preset (DRAFT, sglang)

**Draft from the Model Scout `phonellm-alpha-2` study.** Listed in `catalog.yaml` as `phonellm-mini-alpha-2`; do not merge until the final checkpoint and model card are released.

- Checkpoint: `pipecat-ai/phonellm-mini-alpha-2` @ `4218fdc917a3a56340bf49de7b906637e131d9c1` (final release; main on
  2026-10-15). This preset downloads weights anonymously and requires the pinned revision to be public.
  Every file except README.md (all safetensors shards, `config.json`, tokenizer files, chat template, licenses) is
  byte-identical to rc3 (`daily-co/phonellm-mini-alpha-2-rc3` @ `bcc8f77b1e7b5aa100569aabf34c78b680c71c6b`).
- **Benchmarks were measured on rc3** (same weights as the pin above).
- Served model name: `pipecat-ai/phonellm-mini-alpha-2`.
- Hardware: 1x B200 (`B200`), TP1, BF16 weights.
- Base model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (full-parameter SFT, per the rc3 README).

## Serving contract

BF16, max model length 262144 (the checkpoint's native `max_position_embeddings`; no `--context-length` cap), `qwen3_coder` tool parser with auto tool choice,
streaming OpenAI chat completions at `/v1/chat/completions`, 128 in-flight
sequences (`predict_concurrency` 128), log level WARNING, fp32 Mamba SSM state.

**Thinking OFF** (Mini rc3): the server default is `{"enable_thinking": false}` and there is
**no reasoning parser**. Requests may still override per call.

Engine: **SGLang `v0.5.21-cu130`** (Docker Hub, 2026-10-01; build commit
e00930c). Flags were checked against `python/sglang/srt/arg_groups` at v0.5.21:
`--mamba-ssm-dtype` (choices float32/bfloat16/float16), `--tool-call-parser
qwen3_coder`, `--max-running-requests`, `--context-length`, `--tp-size`,
`--ep-size`, `--default-chat-template-kwargs` (JSON object). `nemotron_h.py`
at v0.5.21 has the latent-MoE path.

`packages/start_sglang.py` execs `sglang.launch_server` and appends
`--default-chat-template-kwargs` with the JSON in
`packages/chat_template_defaults.json` as a single argv element, so the JSON
never passes through shell quoting (the llm/AGENT.md failure mode). Max adds
`--reasoning-parser nemotron_3` (in `parser/reasoning_parser_names.py` at
v0.5.21). Max uses `--tp-size 2 --ep-size 2`, as in NVIDIA's SGLang recipe
for Nemotron 3 Super on B200.

## Study results (Model Scout `phonellm-alpha-2`)

> The study results below were measured with `--context-length 32768`. The preset now serves the checkpoint's native 262144-token context; re-check capacity (KV-cache tokens, running requests, Mamba pool) before relying on those numbers at the new limit.

**Capacity: 128 concurrent conversations** (3-run median; every level 1-128 passed in all three runs, no flips). c=160 flips between runs (0.862 / 0.954) and c=192 fails.

Measured with the multi-turn conversation bench from Model Scout tooling PR #729 (not part of this PR): each level runs that many concurrent copies of Daily's aiewf-eval `aiwf_medium_context` conversation (30 turns, 5 tools, temperature 0, ~14.7K-token prompts by the last turns), refilling conversations until the level has at least 300 successful requests and 120 s. Each conversation gets a unique session line after the shared system prompt, so histories don't share cache but the system prompt does.

The metric is **server-side TTFT**: the serving plane's `vm_model_time_to_first_write` histogram (beefeater, `in_cluster="true"`), queried per level window. Cells are `p50 / p95 ms (share of requests under 600 ms)`. A level passes when that share is at least 0.95; capacity is the highest level with every lower level passing. ✗ marks a failing level. n was 290-8,145 server-side requests per level per run.

| c | SGLang 8 workers (this preset), 3-run median | vLLM `--api-server-count 4`, 3-run median | TRT-LLM snapshot 1024, 1 run | Daily vLLM (client-side, author-reported) |
|---|---|---|---|---|
| 1 | 91 / 116 (0.990) | 85 / 108 (1.000) | 83 / 1782 (0.897) ✗ | 197/210 |
| 8 | 111 / 180 (1.000) | 113 / 186 (1.000) | 114 / 2034 (0.913) ✗ | – |
| 16 | 146 / 213 (1.000) | 148 / 211 (1.000) | 156 / 1865 (0.930) ✗ | – |
| 32 | 171 / 239 (0.996) | 183 / 254 (1.000) | 178 / 2063 (0.934) ✗ | 335/514 |
| 44 | – | – | 196 / 3296 (0.882) ✗ | 576/586 |
| 45 | 182 / 262 (1.000) | 207 / 313 (1.000) | 198 / 3861 (0.888) ✗ | 633/671 |
| 64 | 204 / 326 (1.000) | 261 / 503 (0.989) | 217 / 3498 (0.879) ✗ | 418/779 |
| 80 | 216 / 371 (0.991) | 293 / 657 (0.925) ✗ | – | – |
| 96 | 231 / 389 (1.000) | 354 / 881 (0.811) ✗ | – | – |
| 112 | 246 / 415 (0.989) | – | – | – |
| 128 | 306 / 506 (0.986) | – | – | – |
| 160 | 306 / 874 (0.908) ✗ | – | – | – |
| 192 | 443 / 1469 (0.649) ✗ | – | – | – |

- SGLang 8 workers: runs [38533056196](https://github.com/basetenlabs/model-registry/actions/runs/38533056196), [38542677084](https://github.com/basetenlabs/model-registry/actions/runs/38542677084), [38550730944](https://github.com/basetenlabs/model-registry/actions/runs/38550730944). c=160/192 ran only in the last two (2-run median).
- vLLM comparison: `vllm/vllm-openai:v0.30.0` with `--api-server-count 4` (branch `model-scout/phonellm-alpha-2-vllm` @ `4a082d5`), capacity 64; runs 38511861146, 38517566020, 38524344987. Its c=80 limit is engine-side: from c=32 to c=80 queue plus prefill grew 104 ms against 37 ms for the frontend.
- TRT-LLM comparison: `tensorrt-llm/release:1.3.0rc29` with `kv_cache_config.mamba_state_config.periodic_snapshot_interval: 1024` (branch `model-scout/phonellm-alpha-2-trtllm` @ `28f8a9f`), **no passing level**; run 38542705746. Prefix reuse works (96% of prompt tokens cached) and the engine queue stays under 60 ms mean, but 7-12% of first writes take over 600 ms outside the engine's own TTFT (HTTP, request preprocessing or launcher path; not response postprocessing, which a 4-worker test ruled out).
- Daily's column: their own vLLM Mini benchmark as relayed by Lei Pan (client-side, author-reported, not our measurement). Daily's p95 < 600 ms ceiling was 44 conversations.
- 0 conversation errors and 5/5 functional checks (plain, streaming, tool call, multi-turn tool result, 8 concurrent) in every run.

## Why 8 tokenizer workers

The SGLang baseline's TTFT growth was outside the scheduler (MINI-SGLANG-D1: engine TTFT minus queue and prefill grew with load while both stayed flat), i.e. in the tokenizer-manager/HTTP processes. Going from 4 to 8 workers (`--tokenizer-worker-num 8`) moved the knee from 80 conversations (4-worker 3-run median) to 128. Above 128 the contract's in-flight cap applies.

## Limitations

- **Metric boundary.** TTFT here is time to first byte at beefeater on the serving plane, not at a client. Daily's numbers are client-side and include their network path, so they are a reference, not a like-for-like comparison.
- **Placement.** All runs were on `aws-aps1-prod-1` (Mumbai) B200s. Client-side latency from elsewhere adds the network hop.
- **Weights.** Every number was measured on rc3 (`daily-co/phonellm-mini-alpha-2-rc3` @ `bcc8f77`). The pinned release `4218fdc` has byte-identical weights, config and tokenizer, so the numbers apply unchanged.
- **Contract ceiling.** `predict_concurrency` and `--max-running-requests` are 128 (the serving contract's in-flight limit), so levels above 128 queue before they start.
- **Single region, single replica**, no autoscaling: these are per-replica ceilings.

## Workload

`workload.json` (b10-bench, CI) runs Daily-shaped single-turn cells: aiperf D(14500,36), i.e. ~14.5K-token inputs and ~36-token outputs, at concurrency 1/8/16/32/44/64, 500 requests and 300 s per cell, streaming, no shared prefix (every request misses the cache). It is a cold-prefill stress profile, not the conversation workload above. PR CI `smoke` runs a subset at low concurrency.


The bench tokenizer is the public `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
(README `bench.tokenizer` and each workload bench's `model_tokenizer`):
**benchmark tokenization only; serving uses Mini's own template.** Basis: on
2,486 strings (code, unicode, numbers, whitespace, special tokens, aiperf-style
Shakespeare and random-token prompts; 536k tokens) Mini's tokenizer and the
public one at `bf77c3174f68ad409e1c2aa60daeb46e32d1c606` give identical token ids (100%), with the same
131,072 vocab, BOS `<s>` (1), EOS `<|im_end|>` (11), and no added BOS. Mini's
`tokenizer.json` differs from the public file only by a trailing newline.
Neither override accepts a revision, so aiperf reads the public repo's `main`
(`bf77c3174f68ad409e1c2aa60daeb46e32d1c606` on 2026-10-13); recheck if NVIDIA updates it.

## Checkpoint metadata (rc3 bundle)

From the rc3 handoff bundle at the pinned revision (manifest SHA256 verified
2026-10-13; files not committed here, the repo is private):

- `config.json`: `NemotronHForCausalLM`, `model_type: nemotron_h`, 52 layers
  (23 Mamba / 23 MoE / 6 attention), 128 routed experts + 1 shared, top-6, no
  latent MoE, no MTP, `max_position_embeddings` 262144, `torch_dtype`
  bfloat16, `mamba_ssm_cache_dtype` float32, no `quantization_config`.
  `auto_map` points at remote `configuration_nemotron_h` /
  `modeling_nemotron_h`, so `--trust-remote-code` stays.
- README: BF16 safetensors, `license: bsd-2-clause` (NVIDIA Nemotron Open
  Model License still covers the base work), thinking off, recommended
  `qwen3_coder` tool parser and no reasoning parser. That matches this preset.
- Chat template: Nemotron 3 Nano template with `enable_thinking` defaulting to
  false. Tool calls use `<tool_call><function=NAME><parameter=KEY>` XML, the
  `qwen3_coder` format. Prior-turn reasoning is read only from an assistant
  message's `reasoning_content`.
- `variant.json` keeps BF16 `hf-safetensors-dtypes`; the basis is the config
  dtype and README, not a safetensors header read.
- Still a placeholder: `model.json` parameter count (not in the bundle).