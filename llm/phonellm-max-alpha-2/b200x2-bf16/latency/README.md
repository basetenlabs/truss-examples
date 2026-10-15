# PhoneLLM Max Alpha 2 — latency preset (DRAFT, sglang)

**Draft from the Model Scout `phonellm-alpha-2` study.** Listed in `catalog.yaml` as `phonellm-max-alpha-2`; do not merge until the final checkpoint and model card are released.
License: the final `pipecat-ai/phonellm-max-alpha-2` card and bundled `LICENSE.txt` (2026-10-15) declare OpenMDW 1.1, matching the base model. rc3 was under NVIDIA's evaluation-only license; the pinned release uses OpenMDW 1.1.

- Checkpoint: `pipecat-ai/phonellm-max-alpha-2` @ `e7c4cdb96e2682259f73c6ed979178a5bb7d13cf` (final release; main on
  2026-10-15). This preset downloads weights anonymously and requires the pinned revision to be public.
  All 58 safetensors shards, config.json, tokenizer files and chat_template.jinja are byte-identical to `daily-co/phonellm-max-alpha-2` @ `63f1303`
  ("PhoneLLM upload: nemo-marcus-super35ga1001-packed-v6t1ec1tauflow2pcftgtzd-e2/weights_iter_0000579").
  Against rc3 (`daily-co/phonellm-max-alpha-2-rc3` @ `8038e96faf44322a462e314525c479598daecb0e`): all 58 safetensors shards
  changed (same sizes); `config.json`, tokenizer files and generation config are byte-identical; `chat_template.jinja`
  only adds a newline between a text part and a following image/video part (no effect on text-only requests);
  README and license files changed; the release bundles OpenMDW 1.1 in `LICENSE.txt` (see above).
- **Benchmarks were measured on rc3**, not on these weights (see Limitations).
- Served model name: `pipecat-ai/phonellm-max-alpha-2` (the planned public repo id).
- Hardware: 2x B200 (`B200:2`), TP2, BF16 weights.
- Base model: NVIDIA Nemotron Super 3.5 VL 120B-A12B (rc3 used an early-access build); language backbone only, vision encoder and projector removed (rc3 README).

## Serving contract

BF16, max model length 1048576 (the checkpoint's native `max_position_embeddings`; no `--context-length` cap), `qwen3_coder` tool parser with auto tool choice,
streaming OpenAI chat completions at `/v1/chat/completions`, 128 in-flight
sequences (`predict_concurrency` 128), log level WARNING, fp32 Mamba SSM state.

**Thinking ON** (Max rc3, per Lei Pan 2026-10-13): requests send
`chat_template_kwargs` `{"enable_thinking": true, "truncate_history_thinking": false}`, and the server sets the same values as defaults
so a request that omits them behaves the same. The engine's Nemotron v3
reasoning parser splits `reasoning_content` from `content`. **Open:** whether
each engine preserves earlier-turn reasoning in multi-turn chats
(`truncate_history_thinking: false`) must be validated per engine before
results are compared.

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

> The study results below were measured with `--context-length 32768`. The preset now serves the checkpoint's native 1048576-token context; re-check capacity (KV-cache tokens, running requests, Mamba pool) before relying on those numbers at the new limit.

**Capacity: 64 concurrent conversations, thinking on** (3-run median; no flips). c=80 fails in all three runs (0.815 / 0.872 / 0.869).

Measured with the multi-turn conversation bench from Model Scout tooling PR #729 (not part of this PR): each level runs that many concurrent copies of Daily's aiewf-eval `aiwf_medium_context` conversation (30 turns, 5 tools, temperature 0, ~14.7K-token prompts by the last turns), refilling conversations until the level has at least 300 successful requests and 120 s. Each conversation gets a unique session line after the shared system prompt, so histories don't share cache but the system prompt does.

The metric is **server-side TTFT**: the serving plane's `vm_model_time_to_first_write` histogram (beefeater, `in_cluster="true"`), queried per level window. Cells are `p50 / p95 ms (share of requests under 600 ms)`. A level passes when that share is at least 0.95; capacity is the highest level with every lower level passing. ✗ marks a failing level. n was 290-8,145 server-side requests per level per run.

| c | SGLang 360-slot pool (this preset), 3-run median | SGLang default pool, 3-run median | vLLM bf16 SSM (AMEND-10), 3-run median | TRT-LLM snapshot 1024, 1 run |
|---|---|---|---|---|
| 1 | 157 / 188 (0.976) | 155 / 247 (0.965) | 124 / 155 (0.993) | 124 / 1771 (0.934) ✗ |
| 8 | 255 / 373 (0.997) | 256 / 366 (1.000) | 158 / 254 (1.000) | 220 / 3220 (0.889) ✗ |
| 16 | 285 / 401 (0.993) | 282 / 387 (0.999) | 202 / 342 (1.000) | 249 / 2256 (0.898) ✗ |
| 32 | 304 / 404 (0.983) | 308 / 407 (0.985) | 250 / 521 (0.976) | 281 / 2112 (0.911) ✗ |
| 44 | – | 312 / 416 (0.976) | 279 / 677 (0.924) ✗ | 292 / 2107 (0.924) ✗ |
| 45 | 311 / 415 (0.984) | 302 / 421 (0.978) | 279 / 620 (0.945) ✗ | 305 / 2708 (0.865) ✗ |
| 64 | 331 / 504 (0.974) | 1320 / 2096 (0.075) ✗ | 429 / 1181 (0.658) ✗ | 434 / 4287 (0.729) ✗ |
| 80 | 333 / 925 (0.869) ✗ | – | – | – |

- This preset: runs [38543229245](https://github.com/basetenlabs/model-registry/actions/runs/38543229245), [38551483437](https://github.com/basetenlabs/model-registry/actions/runs/38551483437), [38551499292](https://github.com/basetenlabs/model-registry/actions/runs/38551499292). The server logs `max_running_requests is capped to 72 by the mamba state cache (max_mamba_cache_size=360, 5 state slots per request)`; KV pool 2,245,824 tokens.
- SGLang default pool (same engine, no `--max-mamba-cache-size`): capacity 45; runs 38511866512, 38517576974, 38524356859.
- vLLM comparison: `vllm/vllm-openai:v0.30.0` with `--mamba-ssm-cache-dtype bfloat16` (branch `model-scout/phonellm-alpha-2-vllm` @ `4a082d5`), capacity 32; runs 38521092755, 38560156261, 38560172520. **bf16 SSM state is a contract change** (AMEND-10, approved by Lei Pan for vLLM only); with the contract's fp32 state, vLLM holds 16.
- TRT-LLM comparison: `tensorrt-llm/release:1.3.0rc29`, TP2/EP2, snapshot interval 1024 (branch `model-scout/phonellm-alpha-2-trtllm` @ `ba2223a`), **no passing level**; run 38560187529. Engine queue mean stays at 4-77 ms through c=45, but 7-14% of first writes take over 600 ms outside the engine's TTFT, as on Mini.
- No Daily reference exists for Max concurrency (their lightly loaded figure is ~100 ms TTFT, ~200 ms time to first answer token).
- 0 conversation errors and 5/5 functional checks in every run.

## Why `--max-mamba-cache-size 360`

SGLang v0.5.21 caps running requests at `max_mamba_cache_size // 5` for this hybrid model (3 state slots per request plus 2 for the radix cache's extra buffer with the overlap scheduler; `mem_cache/kv_cache_configurator.py`). With the default `--mamba-full-memory-ratio 0.9` the pool had 221 fp32 slots, so at most 44 requests ran and c=64 queued in the engine (queue mean 1.1 s; the frontend gap stayed ~90 ms). 360 slots raise the cap to 72 and shrink the KV pool from 5.13M to 2.25M tokens, still enough for 80 conversations. MTP speculative decoding (3.5 of 4 draft tokens accepted, identical outputs) and 8 tokenizer workers were tested and did not raise capacity (16 and 45).

## Limitations

- **Metric boundary.** TTFT here is time to first byte at beefeater on the serving plane, not at a client. Daily's numbers are client-side and include their network path, so they are a reference, not a like-for-like comparison.
- **Placement.** All runs were on `aws-aps1-prod-1` (Mumbai) B200s. Client-side latency from elsewhere adds the network hop.
- **Weights.** Every number was measured on the rc3 checkpoint (`daily-co/phonellm-max-alpha-2-rc3` @ `8038e96`), not on the release weights pinned above (`e7c4cdb`, same weights as `63f1303`). Architecture, config and tokenizer are identical, so capacity should carry over, but latency and output length were not re-measured on the new weights.
- **Contract ceiling.** `predict_concurrency` and `--max-running-requests` are 128 (the serving contract's in-flight limit), so levels above 128 queue before they start.
- **Single region, single replica**, no autoscaling: these are per-replica ceilings.
- **Thinking on.** TTFT is the first byte of the stream, which is reasoning content, not the first answer token. Time to first answer token is longer and is not measured here.
- **License and publication.** The rc3 numbers were produced under rc3's NVIDIA evaluation license. The released weights use OpenMDW 1.1; confirm with Daily before publishing rc3-era results.

## Workload

`workload.json` (b10-bench, CI) runs Daily-shaped single-turn cells: aiperf D(14500,36), i.e. ~14.5K-token inputs and ~36-token outputs, at concurrency 1/8/16/32/44/64, 500 requests and 300 s per cell, streaming, no shared prefix (every request misses the cache). It is a cold-prefill stress profile, not the conversation workload above. PR CI `smoke` runs a subset at low concurrency.
Each Max bench request also carries `chat_template_kwargs` `{"enable_thinking": true, "truncate_history_thinking": false}` in
`additional_request_params`. No other registry workload passes a nested object
there; the registry validator accepts it, but the bench platform's passthrough
is unverified, which is why the server defaults match.

### Bench tokenizer (public fallback, client side only)

Per Lei Pan (2026-10-13), the **benchmark client** tokenizes with the public
`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` tokenizer (README `bench.tokenizer` and every bench's
`model_tokenizer`). The server still serves with Max's own pinned tokenizer and
chat template from `pipecat-ai/phonellm-max-alpha-2` (identical to rc3). The bench platform takes
no tokenizer revision, so aiperf loads HF `main`; on 2026-10-13 `main` was
`2dc98e2afe4face0e4ce40972a915c45368bd34a` (tokenizer.json sha256 `623c3456...b5b7`), the
revision Lei approved. If `main` moves, re-check before comparing runs.

Max's tokenizer renames 12 special tokens (ids 18-29: `<image>`, `<img>`,
`<box>`, `<so_start>` ...; `<SPECIAL_18>`...`<SPECIAL_29>` in the public
tokenizer) and otherwise matches it. The synthetic prompts avoid those ids and
spellings by construction: aiperf (b10-bench pin `cd59084`) samples contiguous
token windows from its bundled Shakespeare corpus, which contains no `<`
character and no special token. A local reproduction of this workload (seed 42,
500 prompts per input length, 1,500 in all) found no ids 18-29 and none of the
1,012 special-token spellings of either tokenizer, and Max's tokenizer
(`add_special_tokens=False`) produced token ids identical to the public
tokenizer's for all 1,500 prompts. Details: study state branch,
`checkpoint_evidence.max_bench_tokenizer`.

This fallback covers synthetic text performance only. Do not use it for vision
or special-token tests or for quality claims. Functional tool-call and
multi-turn checks (`functional.json`, read by the PR #729 tooling) run against the server, so they use Max's
real tokenizer and template.

### Token accounting and TTFT (thinking on)

Every bench sets `use_server_token_count: true`, so aiperf requests
`stream_options.include_usage` and takes input/output lengths and token
throughput from the server's `usage`. If the server returns no usage, b10-bench
stamps the row `kind_specific.server_token_counts.applied = false` and the
lengths fall back to client counts; report such rows as client-count fallback.

- **TTFT is time to the first reasoning token.** aiperf stamps TTFT at the first
  streamed chunk whose delta has non-empty `reasoning_content`/`reasoning` or
  `content`; with the reasoning parser on, that is the first reasoning chunk.
  aiperf also computes time to first visible answer token (`time_to_first_output_token`),
  but b10-bench's result row does not carry it.
- **Output tokens include reasoning.** aiperf's output sequence length is
  answer tokens plus reasoning tokens (server: `usage.completion_tokens`), and
  output token throughput and inter-token latency use that total.
  `min_tokens`/`max_completion_tokens` bound the whole completion, reasoning
  included.

## Checkpoint metadata (rc3 bundle)

From the rc3 handoff bundle at the pinned revision (manifest SHA256 verified
2026-10-13; files not committed here, the repo is private):

- `config.json`: `NemotronHForCausalLM`, `model_type: nemotron_h`, no
  `auto_map` (native transformers >= 5 NemotronH; the README asks for
  transformers >= 5.8 or vLLM >= 0.30). 88 layers given as a
  `layers_block_type` list (40 Mamba / 40 MoE / 8 attention), 512 routed
  experts + 1 shared, top-22, `moe_latent_size` 1024,
  `num_nextn_predict_layers` 2 (MTP weights present, not used here),
  `max_position_embeddings` 1048576, `dtype` bfloat16,
  `mamba_ssm_cache_dtype` float32. The config carries no weight-prefix or
  VL-remap hint; the bundle has no safetensors index to confirm tensor names.
- README: `license: other` (NVIDIA Software and Model Evaluation License):
  internal test and evaluation only, not production; authorized users only;
  NVIDIA's prior written consent is needed before disclosing evaluation
  results. Thinking on, `nemotron_v3` reasoning parser and `qwen3_coder` tool
  parser, as this preset uses.
- Chat template: thinking on by default (`enable_thinking` true,
  `truncate_history_thinking` true unless overridden). Tool calls use the
  `qwen3_coder` XML format. Prior-turn reasoning is read only from an
  assistant message's `reasoning_content`. The README tells clients to resend
  it as `reasoning`: vLLM 0.30 and TRT-LLM rc29 copy `reasoning` into
  `reasoning_content`, but SGLang 0.5.21 drops an unknown `reasoning` field and
  needs `reasoning_content`.
- `variant.json` keeps BF16 `hf-safetensors-dtypes`; the basis is the config
  dtype and README, not a safetensors header read.
- Still a placeholder: `model.json` parameter count (not in the bundle).