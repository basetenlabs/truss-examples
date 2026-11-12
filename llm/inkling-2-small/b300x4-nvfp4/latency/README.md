# Inkling 2 Small — SGLang, B300:4, DSPARK

Thinking Machines' Inkling2, served from the public NVFP4 checkpoint
`thinkingmachines/Inkling-2-Small` (revision `b0e7afb8892a07897c78ce1c5b801a0d9cbd5e7b`). The repo is
ungated and Apache-2.0, so no HF token or secret is needed. Before the public release, this preset was
tested on the same weights from Thinking Machines' private early-access repo.

## Checkpoint

From `config.json` and the repo README at the pinned revision:

- `Inkling2ForConditionalGeneration` / `model_type: inkling2` (the early-access checkpoint
  said `inkling2_mm_model`).
- 48 layers, hidden 4608, 48 query / 8 KV heads. Most layers use 1,024-token sliding-window
  attention; every sixth layer (5, 11, …, 47) is global.
- MoE in every layer: 384 routed experts, top-8, no shared experts, expert
  intermediate size 2304.
- `model_max_length` 1,048,576. Text and image input; Inkling's audio path is not
  included.
- Expert weights are packed NVFP4 and MXFP8, as described in `hf_quant_config.json`;
  other parameters keep their declared precision. There is no `quantization_config` in
  `config.json`, so `preset_metadata.py` cannot source the precision. `variant.json` was
  written by hand from `hf_quant_config.json`.
- `mtp.safetensors` holds a separately trained DSPARK draft (5 layers,
  `max_position_embeddings` 262,144).
- H200k tokenizer. The public repo ships `chat_template.jinja`.
- About 394 GB: 24 shards and `mtp.safetensors`.

## Serving

Inkling-2-Small is not in an SGLang release yet (upstream PR
[sgl-project/sglang#50186](https://github.com/sgl-project/sglang/pull/50186)). This preset
uses SGLang's official dev image for it, `lmsysorg/sglang:dev-inkling-2-small@sha256:424c1da97f6d6eef1612006c77b2ca791657a8184c1fbdbc682ebf9485d3b32c`,
as the SGLang Inkling-2-Small cookbook recommends until a release ships.

- **Flags:** the cookbook's B300 NVFP4 recipe. DSpark on B300 is marked "in-progress" upstream, not yet verified.
  - `--tp 4 --ep 4`. NVFP4 experts need EP equal to TP.
  - `--moe-a2a-backend megamoe`: DeepGEMM's fused MoE kernel. It needs
    `SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=4096`, which covers a 16,384-token
    prefill chunk at EP4.
  - `--attention-backend fa4 --bf16-gemm-backend torch --dtype bfloat16 --page-size 128
    --grammar-backend xgrammar`.
  - `--enable-multimodal` turns on image input.
  - `--reasoning-parser inkling --tool-call-parser inkling2`.
  - `--speculative-algorithm DSPARK --max-running-requests 192` turns on the bundled MTP
    draft, with the recipe's request cap. Upstream says the draft covers the full window; this
    preset keeps the 131,072-token cap (`--context-length 131072`) it had with the fork. The
    full-context variant without the draft is `llm/inkling-2-small/b300x4-nvfp4/long-context`.
- **Earlier image:** before the public release, this preset ran Thinking Machines'
  early-access SGLang fork, prebaked with Baseten's `patch_inkling2.py` into
  `docker.io/baseten/baseten-custom-model:inkling2-sglang-0.5.22-fork-f1d0daa7`. That fork
  only accepts the early-access `model_type: inkling2_mm_model` and crash-loops on the
  public checkpoint (`KeyError: 'inkling2'`). The validation results below were measured
  with that fork and its swizzle patch, not with this image.

## Benchmark tokenizer

The benchmark counts tokens with the checkpoint's own tokenizer,
`thinkingmachines/Inkling-2-Small` (`bench.tokenizer` in `workload.json`), so input and
output lengths are exact. The PR CI results below predate this: they used the Inkling v1
tokenizer (`thinkingmachines/Inkling-NVFP4`, o200k merges), so their lengths are approximate.

## Quality

Measured on this exact config (image digest, flags, public checkpoint) deployed on B300:4,
through b10-bench (`run-944ff890c7c1`), temperature 0, reasoning on.

| Eval | Score | Reference |
|---|---|---|
| MMMU-Pro (standard, 10 options; 1,730 questions) | 80.5% (±1.0) | Model card: 81.6% |
| IFEval (541 prompts, prompt-level strict) | 91.7% | — |
| BFCL simple_python (400) | 90.0% | — |
| BFCL parallel (200) | 88.5% | — |
| BFCL multi_turn_base (200) | 41.5% | — |

- BFCL runs through the generic `openai/gpt-oss-120b` handler with native tool calling.
- **Probes passed:** basic completion, reasoning (separate `reasoning_content`, streamed and
  non-streamed), streamed and parallel tool calls, JSON schema, image input, long-decode
  stability, stream termination, and content delivery.
- **DSPARK:** under eval load (32 concurrent requests), the accept length is about 3.9
  tokens per step (accept rate about 0.42).
- MMLU did not complete in the harness, so it is not reported.

## Validation

**PR CI, dev image** (`run-a94b32650105`, GitHub run 43310363123, B300:4, public checkpoint, through Baseten's gateway). Tokens are counted with the checkpoint's own tokenizer, so lengths are exact. Quality on this image is under **Quality** below.

| Workload | Concurrency | P50 time per token | P50 time to first token | Output tok/s |
|---|---|---|---|---|
| 1024/1024 | 1 | 3.9 ms | 92 ms | 261 |
| 1024/1024 | 4 | 4.9 ms | 102 ms | 759 |
| 128/128 | 1 | 3.7 ms | 75 ms | 218 |
| 128/128 | 4 | 5.6 ms | 89 ms | 601 |

### Early-access fork (historical)

Everything in this subsection was measured on Thinking Machines' early-access SGLang fork with Baseten's swizzle patch, before the switch to the public checkpoint and the `dev-inkling-2-small` image. It is kept for reference and does not describe this preset's current image.

**PR CI, early-access fork** (`run-a4e46bb6205a`, B300:4, through Baseten's gateway). Run on the private early-access repo, before the switch to the public checkpoint. Benchmark lengths are approximate, because tokens were counted with the Inkling v1 tokenizer (see above).

| Workload | Concurrency | P50 time per token | P50 time to first token | Output tok/s |
|---|---|---|---|---|
| 1024/1024 | 1 | 16.4 ms | 396 ms | 59 |
| 1024/1024 | 4 | 18.3 ms | 377 ms | 207 |
| 128/128 | 1 | 14.9 ms | 309 ms | 54 |
| 128/128 | 4 | 16.8 ms | 315 ms | 196 |

**Tuning box** (same config, B300:4, benchmarked next to the server, 1024 in / 1024 out):

| Concurrency | P50 time per token | P50 time to first token | Output tok/s |
|---|---|---|---|
| 1 | 8.2 ms | 167 ms | 118 |
| 4 | 10.1 ms | 211 ms | 357 |
| 16 | 14.9 ms | 264 ms | 973 |
| 32 | 19.3 ms | 312 ms | 1,528 |

**KV capacity:** Each GPU holds a 7.94M-token global-attention KV pool and a 0.79M-token sliding-window pool. Running requests are capped at 48 by DSPARK, and all 48 fit at the full 131,072-token context.

**Correctness:** DSPARK's accept length is about 3.0 tokens per step on random-token prompts and about 2.75 on ShareGPT chat. The swizzle patch's greedy outputs are bit-identical to the unpatched fork (checked on B300:8). GSM8K (first 400 test questions, temperature 0) scored 97.5% for the unpatched fork with DSPARK on B300:8. TP4 changes the order of summation relative to TP8, so the exact text differs; answers checked on a 12-prompt greedy set were correct. GSM8K on B300:4 with DSPARK: 97.0% (388/400), 2 questions below B300:8.

**Feature checks** (B300:4 tuning box, same config, temperature 0):

| Check | Result |
|---|---|
| Tool calling | Called `get_weather` with `{"city": "Paris"}` (`finish_reason: tool_calls`), then answered from the tool result |
| Image input | Red/blue two-color PNG: "Red is on the left and blue is on the right." |
| Reasoning | Reasoning returned separately in `reasoning_content`; answer correct ($12) |
| Long context | 118,427-token prompt (90% of the 131,072 limit), hidden fact at 60% depth: found (7.6 s) |