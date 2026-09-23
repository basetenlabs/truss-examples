# Nemotron 3 Diarization (streaming) — validation, step cost, concurrency

Measured on Baseten **fde-internal**, 1× **RTX-PRO-6000** per replica, NeMo `3c2d62ae7eb4`. DER *quality* is
inherited from the model (`../batch/BENCHMARK.md`); this file proves that the live output matches the
whole-file path and measures how many real-time streams one GPU holds. Load comes only from in-cluster
k6 runners (`rsi-bench/nemotron_diar/loadgen/`, scenario `diar_stream_k6.js`); the laptop never generates load.

**Ceiling to compare against.** NVIDIA's reference script (`e2e_diarize_speech.py`, bf16 + `torch.compile`,
all files of a batch stepped in lockstep — offline throughput, not latency-bounded serving) reaches
**RTFx 1,274 at `low` bs=32 and 438 at `ultralow`** on this GPU (eager bf16 bs=32: 735;
`rsi-bench/nemotron_diar/results_nemo_ref/NEMO_REFERENCE.md`). One real-time stream is 1 audio-s/s, so
N held streams = N/1,274 of that ceiling at `low`. The ceilings are bf16, so the fraction is
apples-to-apples for the bf16 build below and understates the fp32 builds.


## 0. General Access checkpoint (2026-09-19) — what was re-measured

Sections 1–5 were measured on the early-access **preview** checkpoint and are kept as history. The GA
checkpoint (retrained bf16 weights, speaker cache 264, same profiles) was validated on this build on
2026-09-21 (`rsi-bench/nemotron_diar/release/GA_VALIDATION.md` §5): load + 16 whole-step graphs per
profile capture unchanged; 20-min in-cluster screens with an idle 1-VU control — **560 `low` holds**
(525/560 completed, 35 gateway drops, 0 rejected, worst 10-min window p95 0.579 s vs control 0.534 s,
p50 growth +0.025 s) and **200 `ultralow` holds** (185/200, p95 0.260 s). `predict_concurrency 560`, the
`low`/`ultralow` admission weights and `concurrency_target` 400 stand. Identity vs the GA batch endpoint
(10 files, DER@0 mean): `low` 2.13, `ultralow` 2.78, `offline` 1.26 — wider than §2's preview band because
the batch preset now runs a bf16 core and this preset fp32; re-measure with matched dtypes before restating
the band. Hour-long arms and the fused-vs-legacy self-check were not re-run on GA.

## 1. The stepper, and what changed in this build

The previous build ran one `forward_streaming_step` per connection and held ~60 `low` streams on 90 s
clips (collapsed at 60 within 20 minutes on hour-long audio). The cross-session stepper (`packages/xsession.py`,
commit `d00e5ac`) batched every connection's ready chunk into one forward per tick and held 250 hour-long
`low` streams — with the stepper thread 94 % busy and the GPU at ~55 %. This build keeps that stepper's
protocol and changes the tick (`NEMO_DIAR_XSESSION_TICK`, default `fused`; `legacy` kept for A/B):

| | `legacy` tick (commit `035414e`) | `fused` tick (this build) |
|---|---|---|
| per-session state | one `StreamingSortformerState` per session; `torch.cat` of B rows per tick, scattered back as views | preallocated **slot tables** per profile (`[640, ...]` fp32); rows gathered/written by slot index inside the graph |
| GPU work per tick | mel (pageable H2D, sync), core replayed from a CUDA graph per B, NeMo `streaming_update_async` eagerly (host sync at the compression decision), D2H | **one CUDA graph per B for the whole step**: mel → gather → pre-encode/concat+pad/FastConformer/head → NeMo's async update with compression computed for every row and selected by `where` (no host sync) → write-back; pinned H2D/D2H, one event |
| pipelining | none: the thread waits for the GPU inside every tick | one tick in flight: tick k+1 is packed and issued while the GPU runs k; the wait is an event sync with the GIL released |
| batch size | drain what is queued (B ≈ 3 at 250 streams) | coalesce to **16 rows** or 40 ms (`NEMO_DIAR_XSESSION_B_TARGET` / `_WAIT_MS`), graphs for B ≤ 32 (`offline` 8) |
| core kernels | fp32 eager | fp32 eager / `torch.compile`'d encoder (`NEMO_DIAR_FUSED_COMPILE`, default 1) / bf16 weights (`NEMO_DIAR_CORE_DTYPE`) |
| handler (event loop) | `json.dumps` head + join of all frozen turn fragments, sorted per partial; partial built in a thread pool; per-frame `np.concatenate` | `TurnTracker` keeps a **committed prefix string** (a partial is O(uncommitted turns); byte-identical to the whole-history serialisation on 26 k random partials); partial built on the loop; frames appended to a list and joined once per chunk; one loop wake-up per tick |

### 1.1 Equivalence
`NEMO_DIAR_BENCH=1` runs a load-time self-check: one synthetic two-talker session stepped chunk by chunk
through the fused graph tick and through the legacy tick with the eager core (NeMo's own
`streaming_update_async`), same model instance. **max |Δ| = 0.0 and identical turns on `low`, `ultralow`
and `offline`** (139 / 417 / 4 chunks; the run covers cache fill, FIFO pops and speaker-cache compression).
The sync-free compression is NeMo's `_compress_spkcache` on every row with device-side indices; rows that do
not need it keep their gathered cache through `where`.

### 1.2 Per-tick budget at 250 `low` streams (10-minute arms, k6 in-cluster, 250 streams each)
Stepper-thread phase timers (`perf_counter` marks, no profiler), from the busiest session-end summary:

| | `legacy` (fp32) | `fused` (fp32 eager) |
|---|---|---|
| rows per tick (B) | **3.17** (B hist 1–8) | **6.9** at 20 ms window (B hist 4–11); 16 at the 40 ms / 16-row default when ≥ 400 rows/s |
| ticks / s | 109 | 50 |
| tick latency p50 / p99 | 8.95 / 15.8 ms | 13.3 / 21.3 ms (includes the coalescing window) |
| thread busy (not blocked on the queue) | **75.6 %** at 10 min (94 % at the hour in the previous ladder — the partial grows) | 67.9 %, of which **37 % is the event wait with the GIL released** and 26 % the coalescing wait |
| GIL-holding Python per tick | mel 1.69 (pageable H2D sync) + post 5.68 (the GPU wait surfacing after the core) + update 1.13 (compression sync) + gather 0.10 + scatter 0.06 + d2h 0.04 + hr 0.17 = **~8.9 ms** | pack 0.55 + issue 0.26 + deliver 0.12 + eager (first chunk / flush) 0.40 = **~1.3 ms** |
| GIL-holding Python per second | ~0.97 core | ~0.07 core |
| GPU per row (CUDA events) | — (not instrumented) | 1.48 ms at B≈7 |
| GPU util (nvidia-smi mean) | 62 % | 49 % |
| emit lag p50 / p95 (k6) | 0.456 / 0.558 s | 0.470 / 0.572 s (+14 ms: the window) |
| finals / transport drops | 222/250, 28 drops | **250/250, 0 drops** |
| event loop: per frame / per partial | 30 µs / 60 µs (8.7 KB partials at 10 min) | same code path (measured on the legacy run: `loop_busy_frac` 0.068, process 0.86 cores) |

Reading: the legacy tick's 8.9 ms is almost entirely the thread waiting on the GPU while holding the GIL
(the H2D from pageable memory, the mask/downsample after the core, and NeMo's `if len(idx)` compression
test each force a stream sync), so ticks stayed small (B ≈ 3) and the GPU ran batch-3 kernels at ~1.7 ms/row.
The fused tick has no sync until its device-to-host copy, waits on an event, and coalesces to bigger B.
Raw logs: `rsi-bench/nemotron_diar/results_stream/xs_ladder2/{legacy_prof,fused_prof}/N250/`.

### 1.3 Fused tick GPU cost per batch size (`low`, steady geometry, graph replay, load-time bench)

| B | fp32 eager kernels | fp32 + compiled encoder (`NEMO_DIAR_FUSED_COMPILE=1`) | **bf16 core + compiled** (`NEMO_DIAR_CORE_DTYPE=bf16`) | fp32 + compiled, SDPA attention (`NEMO_DIAR_ATTN=sdpa`) |
|---|---|---|---|---|
| 1 | 3.41 ms | 2.64 | 2.03 | 4.53 |
| 4 | 6.63 (1.66 /row) | 5.31 (1.33) | 3.51 (0.88) | 7.74 (1.94) |
| 8 | 10.94 (1.37) | 8.47 (1.06) | 5.49 (0.69) | 13.52 (1.69) |
| **16** | 20.09 (**1.256**) | 15.24 (**0.953**) | 8.86 (**0.554**) | 25.48 (1.59) |
| 32 | 42.79 (1.337) | 34.40 (1.075) | 17.31 (0.541) | — |

`NEMO_DIAR_ATTN=sdpa` (the batch preset's knob: `F.scaled_dot_product_attention` with NeMo's dense
Transformer-XL bias + the padding mask as one additive `attn_mask`, `create_block_mask` replaced by a
dense mask from the same `mask_mod`) is **1.67× slower** here: a dense `(B, H, T, T)` float mask rules
out the flash kernel and every layer materialises it. DER 18.26 / 12.49 (in band) with a wider per-file
spread (median 0.13, 10 files > 1 pt) and the fused-vs-legacy self-check no longer bit-exact
(max |Δ| 1.4e-2). It stays an A/B knob; FlexAttention is the default.

Per-row cost bottoms out at B = 16 (B = 32 is 11 % worse for fp32, equal for bf16), hence the 16-row
coalescing target and, since the GPU-bound `ultralow` 250 arm ran every tick at the 32-row cap, a 16-row
graph cap (`NEMO_DIAR_XSESSION_B` 16). Implied GPU-only ceilings at `low` (0.72 s of audio per row): fp32 eager 573 rows/s
(= streams), compiled 755, bf16 1,300 — i.e. the bf16 fused tick reaches NVIDIA's lockstep ceiling
(1,274) per row; what is left is the coalescing window (B < 16 below ~400 streams) and the event loop.

## 2. Correctness

### 2.1 Streaming vs the whole-file (batch endpoint) path, 10 NOTSOFAR files, DER @collar 0 (previous ladder)
The batch endpoint runs NeMo's *sync* streaming schedule on the whole file; every build here runs the
*async* fixed-shape schedule (same cache/FIFO update rule with per-row lengths), so the residual is the
schedule plus kernel rounding, and the model amplifies rounding through its speaker-cache decisions.

| file | previous build (compile) | **stepper, idle (B=1)** | stepper while sharing ticks (B 1–17) | sync graphs (`NEMO_DIAR_SYNC_GRAPHS=1`) |
|---|---|---|---|---|
| MTG_32000 | 0.33 | 0.34 | 0.34 | 0.34 |
| MTG_32003 | 2.12 | 0.24 | 0.24 | 0.23 |
| MTG_32004 | 0.96 | 0.22 | 0.22 | 0.21 |
| MTG_32005 | 0.46 | 0.13 | 0.13 | — |
| MTG_32006 | 0.44 | 0.12 | 0.13 | — |
| MTG_32007 | 0.65 | 0.40 | 0.68 | — |
| MTG_32008 | 1.16 | 0.29 | 1.15 | — |
| MTG_32009 | 1.39 | 1.34 | 1.36 | — |
| MTG_32020 | 1.99 | 0.36 | 0.68 | — |
| MTG_32021 | 2.22 | 0.16 | 0.16 | — |
| **mean / max** | **1.17 / 2.22** | **0.36 / 1.34** | **0.51 / 1.36** | 0.26 / 0.34 (3 files) |

- Idle, the stepper is closer to the batch path than the previous build (mean 0.36 vs 1.17): its eager
  fp32 kernels are the batch endpoint's kernels. MTG_32009 (1.34) is the async-vs-sync schedule itself.
- Batch composition matters at the rounding level: rows sharing a tick run the B-row kernels, which round
  differently from the B=1 kernels, and on some files the speaker cache amplifies that. It is intrinsic to
  batching this model (pinned row counts did not remove it); per-row exactness is available only unbatched
  (`NEMO_DIAR_SYNC_GRAPHS=1`).
- Other profiles, stepper idle (B=1), 10 files: **`ultralow` mean 1.00 / max 2.59**, **`offline` mean 0.19 /
  max 0.65** (the `ultralow` residual is the async schedule at the smallest chunk).
- This build's fused tick is bit-identical to that stepper at equal batch composition (§1.1), so these
  residuals carry over for the fp32-eager core. The compiled and bf16 cores round differently; their
  10-file identity (`ultralow`, `low`) is in §2.4.

### 2.2 Full-set streaming DER, NOTSOFAR-129 at `low`, 6 concurrent streams (rows share ticks)
`stream_bench_q.py` through the WebSocket, finals scored against the references. Acceptance: within 0.3 of
the previous build's 18.09 / 12.31.

Per-file stability = |ΔDER@0| per file against the stepper-v1 run (same 129 files, same 6-wide protocol):

| build | DER @0 | DER @.25 | per-file mean / median / p90 abs Δ | files > 1 pt | worst file | notes |
|---|---|---|---|---|---|---|
| previous per-connection build (committed, `results_stream/notsofar_stream_low`) | 18.24 | 12.47 | | | | |
| stepper v1 (`035414e`), 6 streams sharing ticks | 18.09 | 12.31 | reference | | | the band |
| **fused, fp32 eager core**, 20 ms window (`xs2_notsofar_low`) | 18.27 | 12.50 | 0.29 / 0.06 / 0.62 | 4 | MTG_32055 30.4 → 38.1 | |
| **fused, fp32 + compiled encoder**, 20 ms window (`xs2c_notsofar_low`) | 18.18 | 12.41 | 0.30 / 0.05 / 0.45 | 4 | MTG_32026 18.4 → 27.6 | |
| **fused, fp32 + compiled, 16-row / 40 ms coalescing = shipped** (`xs2f_notsofar_low`) | 18.07 † | 12.31 † | 0.16 / 0.03 / 0.42 | 2 | MTG_32026 18.4 → 21.1 | † 126 of 129 files: 3 sockets were reset by the gateway before the final (the server logged 129 complete sessions of ~358 s); the other builds score 18.13 / 12.38 (compiled) and 18.22 / 12.48 (eager) on the same 126. The harness now retries a socket that closes without a final. |
| fused, fp32 + compiled, 16-row / 40 ms coalescing, **min 2 rows per tick = shipped** (`xs2m2_notsofar_low`) | 18.25 | 12.47 | 0.36 / 0.05 / 0.75 | 6 | MTG_32026 18.4 → 27.2 | 6-wide ticks never used the 1-row graph, so this equals the row above within composition noise; the pad matters for lone sessions (§2.4) |
| fused, fp32 + compiled, SDPA attention (`xs2sdpa_notsofar_low`, A/B) | 18.26 | 12.49 | 0.34 / 0.13 / 0.68 | 10 | MTG_32055 30.4 → 38.1 | 1.67× slower per row (§1.3); not shipped |
| fused, bf16 core + compiled (`xs2bf16_notsofar_low`) | 18.15 | 12.37 | **0.58 / 0.26 / 1.33** | **19** | MTG_32180 16.8 → 23.4 | set-level inside the band, per-file spread 2× the fp32 cores (bf16 vs fp32-compiled head-to-head: mean 0.59, max 8.7) |
| fused, bf16 core + compiled, cuBLAS reduced-precision reductions **off** (`xs2bf16r_notsofar_low`) | 18.26 | 12.47 | 0.65 / 0.22 / 1.81 | 16 | MTG_32055 30.4 → 40.7 | the split-K accumulation hypothesis for the spread does not hold: same spread with bf16-accumulated reductions disabled (`NEMO_DIAR_REDUCED_PRECISION_REDUCTION=0`, now the default for every core) — the spread is bf16 rounding itself |
| fused, bf16 core + compiled, **min 8 rows per tick** (`xs2bf16m8_notsofar_low`) | 18.22 | 12.45 | 0.65 / 0.22 / 1.58 | 21 | MTG_32180 16.8 → 23.5 | the bs=1-graph hypothesis (§2.4) does not explain the bf16 spread either: every tick of this run had ≥ 8 rows and the spread is the same. bf16 stays behind `NEMO_DIAR_CORE_DTYPE=bf16` |

All builds are inside the 0.3 set-level band. The fp32 cores' per-file spread (median 0.03–0.06, a handful of
files > 1 pt) is the batch-composition rounding of §2.1 — which rows share a tick changes with the
coalescing rule, not the kernels. bf16 is a different rounding draw on every file (median 0.26, 19 files
> 1 pt) with the same set-level DER; by the per-file criterion it is not shipped as the default core.

### 2.3 Mixed-offset run (previous ladder): 16 different files, real time, staggered 0.7 s, vs each file's own N=1 final
14/16 finals, none byte-identical to its N=1 final; DER(vs own N=1)@0 ≤ 0.25 on 8 files, 0.5–0.74 on 4,
6.35 and 10.5 on 2 (composition effect including two speaker-map flips). Unchanged by this build (same
kernels at equal composition).

### 2.4 10-file identity vs the batch endpoint, compiled / bf16 cores (`xs_correctness.py identity`)
DER(batch endpoint's whole-file output vs the streamed final) @collar 0, one stream at a time — so every
tick of these runs was a **1-row tick** and, on the compiled cores, replayed Inductor's bs=1-specialised graph.
Acceptance for a default core: `ultralow` within the committed build's band (mean 1.00 / max 2.59).

| core | rows per tick | `low` mean / max | `ultralow` mean / max | verdict |
|---|---|---|---|---|
| fp32 eager (stepper v1 = this build's fused tick at B=1) | 1 | 0.36 / 1.34 | 1.00 / 2.59 | band |
| fp32 + compiled encoder | 1 | 1.17 / 2.22 | 0.84 / 2.54 | band on `ultralow`; `low` +0.8 over eager — the bs=1 compiled graph (below) |
| **fp32 + compiled encoder, min 2 rows (shipped)** | 2 (lone session padded) | 1.12 / 6.06 | 1.10 / 2.45 | `ultralow` mean at the band's edge, max inside; the same +0.8 drift over eager as at 1 row — in this path the pad does not remove it (see below) |
| bf16 + compiled (cuBLAS reduced-precision reductions on or off: identical to the digit) | 1 | 2.60 / 8.17 | 3.01 / 8.87 | outside the band |
| bf16 + compiled, min 8 rows | 8 | 2.85 / 10.46 | 2.87 / 8.10 | outside the band — unchanged by the row count: the bf16 spread is the dtype |
| fp32 + compiled, SDPA attention (A/B) | 2 | 1.26 / 2.69 | 2.05 / 6.77 | outside the band on `ultralow`; and 1.67× slower (§1.3) — not shipped |

**The mechanism behind "compile drift".** The batch preset's fork isolated it on the same model and
container: a `low` bf16 file run alone through the compiled bs=1 graph scores +0.65 DER against the eager
path (miss +0.78, one-sided); the *same* file padded into the bs=2 graph scores −0.13 (miss +0.01). TF32 at
bs=1 shows the identical bias, so it is the kernels Dynamo/Inductor selects for a batch of one — not the
dtype — and it is also NeMo's own +0.82 "compile drift" on this model. Forcing fp32 reductions
(`allow_bf16_reduced_precision_reduction=False`) changes nothing (§2.2). Consequences for this truss:
every compiled core now pads a tick below `NEMO_DIAR_XSESSION_B_MIN` (default **2**) rows with idle
scratch slots, so the 1-row graph is never replayed — a lone session costs one extra row (≈ +1 ms per
tick at B=1, irrelevant at one session); the NOTSOFAR-129 6-wide gates ran ticks at 2–6 rows and were
never exposed to it, which is why they sat inside the band while the single-stream identity runs moved.
What the pad did and did not do here: bf16's per-file spread survives it unchanged (F6 rows: 8-row ticks,
same 2.9 mean identity, same 21 files > 1 pt on the 129-file set), and the fp32 compiled core's identity
drift over eager (≈ +0.8 on the 10-file set) is the same at 2 rows as at 1. Our per-B CUDA graphs are
captured from a `torch.compile(dynamic=True)` encoder, so the B=2 graph is the general dynamic-batch
kernel set, not a bs=2 specialisation — the drift that remains is Inductor's fusions versus the eager
(batch-endpoint) kernels, amplified by the speaker cache on a few files (MTG_32003 in every run), and it
does not show on the reference-scored set (§2.2: eager and compiled cores have the same per-file spread
against the stepper-v1 hypotheses). The 2-row default is kept as a no-cost guard against the bs=1 path.

A first attempt at the identity run on a build with a *fixed* 40 ms coalescing window returned 7 of 10
`ultralow` files empty: with one live stream the stepper waited the whole window on every chunk (40 ms ×
1,500 chunks per file), the harness's non-real-time flood then sat buffered at the gateway for minutes and
7 sockets were reset before the final (the server logged all 20 sessions complete). Two fixes came out of
it: the coalescing target is `min(16, live sessions)` — a lone stream never waits (B=1 step 40 → 4 ms; the
20-file run 3 min instead of 40) — and both harnesses retry a socket that closes without a final.

## 3. Concurrency — hour-long real-time ladder (k6 in-cluster, 60 min per stream, 30 s ramp)
`emit lag` = k6 receive time of a partial minus the send time of the 100 ms frame that completed its
`processed_s` (so it includes the profile's right-context buffer: 0.32 s at `low`); window = 10 min of
audio; **degraded** = p95 emit lag > 1.5 s in any window, or end-backlog excess > 0.5 s, or p50 growth
first→last window > 0.5 s, or any unexplained incomplete stream. A 1-VU control runs on an idle replica of
the same build during every arm. Gateway resets (`write: broken pipe`, close 1001; no server-side error)
are counted separately as transport drops. Arms above 250 streams run on deployments with
`predict_concurrency: 800` (truss admits WebSocket sessions through that semaphore; an early N=480 arm on a
250-cap replica showed exactly 230 streams that never received a frame). One 16-vCPU k6 runner saturates
near 400 real-time streams (320 streams: 9.4 cores mean / 11.3 max; a 480-stream run pegged all 16 cores and
k6's own send drift reached p95 1,050 s — the server starved at ~345 rows/s with the GPU at 31 %), so arms
≥ 480 are split over two runners (`lg_arms.py --runners`), each arm's frames concatenated for analysis;
the generator's cores are recorded per arm. Raw per-frame logs, k6 summaries, generator CPU and
`MANIFEST.json` per arm: `rsi-bench/nemotron_diar/results_stream/xs_ladder2/` (this build),
`.../xs_ladder/` (stepper v1).

### 3.1 Stepper v1 (`035414e`, legacy tick), for reference
| profile | N | done (drops) | emit lag p50 / p95 / p99 | window p95 (6 × 10 min) | tick p50 / p99 | GPU mean | verdict |
|---|---|---|---|---|---|---|---|
| low | 160 | 142/160 (18) | 0.53 / 0.64 / 0.66 | 0.55 … 0.65 | 7.1 / 11.9 ms | 56 % | holds |
| low | 200 | 145/200 (55) | 0.53 / 0.64 / 0.67 | 0.56 … 0.66 | 9.5 / 15.8 | 53 % | holds |
| low | 250 | 177/250 (73) | 0.48 / 0.59 / 0.62 | 0.54 … 0.61 | 11.2 / 23.7 | 55 % | holds — edge: stepper thread 94 % busy |
| ultralow | 60 | 41/60 (19) | 0.20 / 0.27 / 0.28 | 0.24 … 0.27 | 10.1 / 16.2 | 45 % | holds |
| ultralow | 100 | 91/100 (9) | 0.22 / 0.30 / 0.32 | 0.25 … 0.32 | 22.9 / 35.5 | 60 % | holds — B at the 16 cap |

The per-connection build before it collapsed at 60 within 20 minutes (window p95 23 → 198 s) and at 80 (0/80).

### 3.2 This build — `low` (fp32 + compiled encoder, coalescing to min(16, live) rows / 40 ms; control = idle replica of the same build)
Columns: done = finals received / streams (gateway resets); emit lag p50 / p95 / p99 over the hour; p95 per
10-min window; end backlog p95 (excess over the 1.04 s chunk buffer); tick latency p50 / p99 at the arm's end;
B = rows per tick over the arm; GPU ms per row (CUDA events, last 4 k ticks); GPU util (nvidia-smi mean);
stepper-thread GIL-holding Python = ms per tick × ticks/s.

| N | done (resets) | emit lag p50 / p95 / p99 | window p95 (6 × 10 min) | end backlog p95 (excess) | tick lat p50 / p99 ms | B | GPU ms/row | GPU util | stepper GIL | control 1-VU p50 / p95 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 250 † | 222/250 (20) | 0.56 / 0.67 / 0.70 | 0.59 / 0.62 / 0.67 / 0.67 / 0.68 / 0.69 | 1.44 (0.4) | 15.2 / 22.0 | 12.3 | 1.11 | 33.8 % | 0.65 ms × 25.6/s = **0.017 core** | 0.56 / 0.65 | **holds** (242 admitted) |
| **320** | 301/320 (19) | 0.56 / 0.67 / 0.69 | 0.58 / 0.62 / 0.66 / 0.67 / 0.68 / 0.69 | 1.44 (0.4) | 17.7 / 27.1 | 15.2 | 1.03 | 43.6 % | 0.81 ms × 28.6/s = 0.023 core | 0.56 / 0.68 | **holds** — identical to the control |
| **400** | 328/400 (72) | 0.56 / 0.68 / 0.72 | 0.58 / 0.62 / 0.67 / 0.68 / 0.70 / 0.71 | 1.44 (0.4) | 18.2 / 30.3 | 15.4 | 1.04 | 48.8 % | 1.50 ms × 31.8/s = 0.048 core | 0.49 / 0.59 | **holds** (72 gateway resets = 18 %; p50 +0.07 s over the control and +0.11 s over the hour — the outbound partials reach 65 KB × 555/s) |
| **480** ‡ | 418/480 (62) | 0.55 / 0.67 / 0.69 | 0.58 / 0.62 / 0.66 / 0.66 / 0.68 / 0.68 | 1.44 (0.4) | 19.4 / 30.2 | 16.2 | 1.01 | 62.3 % | 1.94 ms × 37.9/s = 0.073 core | 0.56 / 0.68 | **holds** — identical to the control (62 gateway resets = 13 %) |
| **560** ‡ | 473/560 (87) | 0.56 / 0.67 / 0.71 | 0.58 / 0.64 / 0.67 / 0.67 / 0.69 / 0.70 | 1.44 (0.4) | 20.7 / 53.0 | 16.2 | 1.01 | 70.3 % (75 % at full load) | 2.86 ms × 40.5/s = 0.116 core | 0.56 / 0.66 | **holds** — p50 = control; the edge (87 gateway resets = 16 %; tick p99 53 ms) |
| 640 ‡ | 398/640 (242) | 0.81 / 89.5 / 260 | 2.20 / 3.95 / 3.37 / 0.80 / 0.71 / 271.9 | 364 (363) | server step 54 / 72 | 32 (cap) | ~1.1 | **92 %** (96–99 % in the first 30 min) | — | 0.56 / 0.66 | **collapse — GPU-bound** (890 rows/s needed ≈ the whole GPU); windows 4–5 recover only because 242 gateway resets thinned the load to ~400 streams |

‡ split over two k6 runners (240 + 240 streams, 7.7 / 9.2 cores each).
† run on a replica still configured with `predict_concurrency: 250` that had 8 lingering sessions from
earlier reset sockets: VUs 243–250 never got a frame (blocked at truss's admission semaphore, 0 frames sent),
so 242 streams ran; every admitted stream held. Later arms run on `predict_concurrency: 800` replicas.
The lag spikes (168 frames > 1.5 s, in clusters at t ≈ 320 s and t ≈ 1,700 s) are gateway hiccups: the
`ultralow` arm on another replica, started at the same wall time, shows the same cluster at t ≈ 1,702 s.

#### 3.2b Re-test of 560 on the shipped build (`049052b`, committed config, admission budget on)
Fresh target + a second replica of the same build as the 1-VU control; 280 + 280 VUs from two runners
(5.9 cores mean / 7.3 max each); every VU starts at its own offset into the hour of audio (280 distinct
offsets, quartiles 7 s / 856 s / 1,706 s / 2,615 s / 3,592 s), every socket close logged with its position.

| | value |
|---|---|
| emit lag p50 / p95 / p99 | 0.55 / 0.67 / 0.70 s — control (idle replica, same build) 0.55 / 0.65 / 0.66 |
| window p95 (6 × 10 min) | 0.58 / 0.63 / 0.66 / 0.67 / 0.69 / 0.69 — control 0.57 / 0.64 / 0.66 / 0.66 / 0.64 / 0.64 |
| end backlog p95 (excess) | 1.44 (0.4) |
| tick lat p50 / p99, B, stepper GIL | 18.0 / 25.1 ms, 15.8 rows, 0.73 ms × 34.8/s = 0.025 core |
| admission | 560 / 560 admitted, 0 rejected (the budget equals the arm) |
| finals | **345 / 560**; 0 unexplained; 345 distinct final outputs (7–8 speakers, median 2,335 turns) |
| gateway resets | **215 (38 %)**, close 1001 (214) / 1006 (1), in three bursts — 104 in minutes 0–5, 116 in 5–10, 122 in 20–25 — then a trickle; at the moment of reset the streams were current (backlog p75 1.1 s, one outlier 30 s); no server-side error |
| verdict | **holds on the latency rule** (every window ≤ 0.69 s, p50 = control, backlog excess 0.4). The completion rate is the gateway's: the first two bursts hit while all 560 were live (lag spike max 2.0 s across 233 streams at t = 25 s, the only cluster), and after minute 25 the replica carried ~345 streams. |

Reading: the server side of 560 is confirmed (latency identical to an idle replica, stepper at 0.025 core,
no errors, no rejections); the platform side is not — at ≥ 480 concurrent WebSocket sessions per replica the
gateway resets a growing share (13 % at 480, 16 % at 560 in the ladder, 38 % here) in bursts unrelated to
server load. Until the delta-partial protocol (§3.5) or the gateway behaviour changes, a listing should
advertise `concurrency_target 400` as the operating point and treat 560 as the admission ceiling.
Raw: `rsi-bench/nemotron_diar/results_stream/xs_ladder2/final560/` (+ `scan560.py` / `N560/scan560.json`).

### 3.3 This build — `ultralow` (fp32 + compiled; 0.24 s chunks, 4.2 partials/s per stream)
| N | done (resets) | emit lag p50 / p95 / p99 | window p95 (6 × 10 min) | end backlog p95 (excess) | tick lat p50 / p99 ms | B | GPU ms/row | GPU util | stepper GIL | control 1-VU p50 / p95 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | 69/100 (31) | 0.22 / 0.28 / 0.30 | 0.24 / 0.27 / 0.28 / 0.29 / 0.29 / 0.29 | 0.48 (0.16) | 14.2 / 24.2 | 11.2 | 1.11 | 35.3 % | 0.60 ms × 29.5/s = 0.018 core | 0.22 / 0.29 | **holds** (31 gateway resets, in bursts) |
| **160** | 145/160 (15) | 0.22 / 0.29 / 0.30 | 0.24 / 0.28 / 0.29 / 0.29 / 0.30 / 0.30 | 0.48 (0.16) | 19.2 / 30.0 | 16.2 | 0.99 | 61.8 % | 1.97 ms × 38.9/s = 0.076 core | 0.22 / 0.29 | **holds** — identical to the control |
| **200** | 149/200 (51) | 0.22 / 0.29 / 0.31 | 0.26 / 0.28 / 0.29 / 0.29 / 0.30 / 0.30 | 0.48 (0.16) | 20.1 / 31.4 | 16.4 | 1.00 | 76.4 % (81 % at full load) | 1.63 ms × 47.6/s = 0.078 core | 0.22 / 0.30 | **holds** — identical to the control (51 gateway resets = 26 %, thinning the load over the hour) |
| 250 | 217/250 (33) | 149.6 / 174.4 / 174.7 | 70.6 / 113.3 / 151.3 / 169.5 / 174.1 / 174.7 | 175.0 (174.6) | 70.7 / 71.1 | 32.0 | 1.11 | **99.5 %** | 8.8 ms × 28.2/s = 0.25 core | 0.21 / 0.23 | **collapse — GPU-bound**: every tick at the 32-row cap, ~914 rows/s delivered vs 1,042 needed, backlog grows linearly all hour |
| 300 | not run (the 250 collapse is the GPU ceiling) | | | | | | | | | | |

### 3.4 bf16 core (+ compiled) — not the default (per-file spread, §2.2); measured for the headroom
| profile | N | done (resets) | emit lag p50 / p95 / p99 | window p95 | end backlog p95 (excess) | tick lat | B | GPU ms/row | GPU util | stepper GIL | control | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| low | 480 ‡ | 371/480 (109) | 0.54 / 0.66 / 0.69 | 0.57 / 0.62 / 0.65 / 0.66 / 0.67 / 0.68 | 1.44 (0.4) | 11.9 / 19.7 | 15.9 | 0.60 | **35 %** (fp32 core: 62 %) | 0.78 ms × 36.9/s = 0.029 core | 0.56 / 0.66 | **holds** (109 gateway resets = 23 %) |
| low | 560 ‡ | 509/560 (51) | 0.55 / 0.66 / 0.70 | 0.57 / 0.62 / 0.65 / 0.66 / 0.67 / 0.69 | 1.44 (0.4) | — | — | — | **42 %** (fp32 core: 75 %) | — (stepper log not sampled for this arm) | 0.56 / 0.67 | **holds** — p50 = control; ~1.8× the fp32 core's GPU headroom (51 gateway resets = 9 %) |
| low | 640 | not run: the fp32 core's 640 arm and the bf16 `ultralow` arms already place the bf16 GPU ceiling above 1,000 low-equivalents; the event loop and the gateway would be the next limits | | | | | | | | | | |
| ultralow | 200 | 185/200 (15) | 0.21 / 0.28 / 0.29 | 0.23 … 0.29 | 0.48 (0.16) | 12.6 / 22.1 | 16.2 | 0.57 | 44.9 % | 1.78 ms × 50.5/s = 0.09 core | 0.21 / 0.26 | **holds** |
| ultralow | 250 † | 211/250 (35) | 0.21 / 0.28 / 0.30 | 0.24 … 0.29 | 0.48 (0.16) | 14.2 / 22.1 | 16.3 | 0.56 | 51.7 % | 2.33 ms × 57.6/s = 0.13 core | 0.22 / 0.22 | **holds** (246 admitted: 4 VUs blocked at the 250-cap replica's admission, 0 frames) |

### 3.5 Sizing (fp32 + compiled encoder, the shipped core)
- **`low`: 560 hour-long real-time streams hold on one RTX-PRO-6000** (p50 emit lag equal to the idle
  control, every window p95 ≤ 0.70 s, GPU 75 % at full load); 640 collapses GPU-bound. Stepper v1 held 250
  with its thread 94 % busy — this build's stepper holds the GIL for ~0.1 core at 560, and the ceiling is
  the GPU (1.0 ms per row at B = 16, i.e. ~890 rows/s ≈ 640 streams at 100 %).
- **`ultralow`: 200 hold** (GPU 76–81 %); 250 collapses GPU-bound (needs 1,042 rows/s ≈ 104 % at 1.0 ms/row).
- **`predict_concurrency: 560`** — the measured `low` hold; excess connections are rejected rather than
  dragging every stream into collapse. **`concurrency_target: 400`** for autoscaling (400 and 480 hold at
  the control's latency; the 160-stream band to 560 absorbs bursts and gateway reconnects). A mostly-
  `ultralow` deployment is 3× the density per stream — and since `predict_concurrency` cannot see the profile
  mix, the server admits sessions against one GPU budget in `low`-row-equivalents (`NEMO_DIAR_MAX_LIVE_LOW`
  560; `ultralow` weighs 560/`NEMO_DIAR_MAX_LIVE_ULTRALOW` = 2.8, `offline` 0.2) and rejects an over-budget
  handshake with an error frame and a clean close (smoke: §3.6).
- **Against the bars:** `low` 560 ≥ 300 (+124 %; 480 ≥ 300 identical to the control), `ultralow` 200 ≥ 120
  (+67 %). **Against NeMo's lockstep ceiling** (bf16 + compile, 1,274 audio-s/s `low`, 438 `ultralow`, the
  same GPU): 560/1,274 = **44 %** and 200/438 = **46 %** — with an **fp32** core, under the real-time
  latency rule, through the WebSocket path, on hour-long audio; the GPU itself sat at 75–81 %. The bf16
  core measures 0.55 ms/row (the reference's own precision: ≈ 1,300 rows/s ≈ NeMo's ceiling per row) and
  held `ultralow` 250 at 52 % GPU, but it doubles the per-file DER spread (§2.2), so it ships as a knob.
- Where the next streams come from (in order): the bf16 core if its per-file spread is acceptable to the
  customer (560 `low` at 42 % GPU vs 75 % — ≈ 1.8× rows/s); a delta
  partial protocol (the replace-style partial reaches 65 KB × 555/s at 400 streams and is the likely
  trigger of the gateway resets that thin every arm); two replicas.

### 3.6 Admission smoke: 260 `ultralow` sessions against a 200-`ultralow` budget (10 min, cloud k6)
`NEMO_DIAR_MAX_LIVE_LOW 560`, `NEMO_DIAR_MAX_LIVE_ULTRALOW 200` (weight 2.8), `predict_concurrency 800` so the
profile budget is the binding cap; k6 opens 260 `ultralow` sessions over a 30 s ramp:

| opened | admitted (finals) | rejected | transport drops | unexplained | admitted streams: worst window p95 | GPU |
|---|---|---|---|---|---|---|
| 260 | **200 / 200** | **60**, each with `{"type":"error","error":"capacity: this replica is at its GPU budget for ultralow sessions (560/560 low-equivalent streams live, ultralow counts 2.8); retry on another replica"}` and a clean close | 0 | 0 | 0.242 s | 79.5 % |

Server log per rejection: `capacity: rejected ultralow session (560/560 low-equivalents live: {'ultralow': 200})`.
Raw: `rsi-bench/nemotron_diar/results_stream/xs_ladder2/admission_smoke/N260/`.

## 4. Cold start
| build | `load()` | notes |
|---|---|---|
| previous (compile, 3 profiles) | 69 s | offline compile 62 s; b10cache saves ~half on restarts |
| stepper v1 (core graphs) | 41 s | 16 core graphs per profile; 12.5 GB reserved |
| **fused, fp32 eager** | 52 s | 32 + 32 + 8 whole-step graphs (2.7 s per profile), slot tables; 27.8 GB reserved (shared graph pool per profile) |
| **fused + compiled encoder** | 132–157 s cold (offline compile ~100 s; low/ultralow reuse the dynamic graph) | b10cache saves the Inductor cache (3.8 MB) for restarts; 22.6 GB reserved |
| fused + compiled, bf16 core | 132 s cold | 15.7 GB reserved |

`startup_threshold_seconds` 600.

## 5. Knobs
`NEMO_DIAR_XSESSION` (1), `NEMO_DIAR_XSESSION_TICK` (fused | legacy), `NEMO_DIAR_CORE` (graphs | eager),
`NEMO_DIAR_FUSED_COMPILE` (1), `NEMO_DIAR_CORE_DTYPE` (fp32 | bf16), `NEMO_DIAR_XSESSION_B` (32, graphs per
size at load; `_B_OFFLINE` 8), `NEMO_DIAR_XSESSION_B_TARGET` (16, stop coalescing), `NEMO_DIAR_XSESSION_WAIT_MS`
(40, window cap), `NEMO_DIAR_XSESSION_SLOTS` (640), `NEMO_DIAR_TF32` (0), `NEMO_DIAR_SYNC_GRAPHS` /
`NEMO_DIAR_COMPILE` / `NEMO_DIAR_ASYNC_STATE` (A/B paths when `NEMO_DIAR_XSESSION=0`), `NEMO_DIAR_PROF`
(per-step records + GPU samples in every frame; timers only), `NEMO_DIAR_PROF_SYNC` (legacy tick: sync after
the core so its phase is GPU time), `NEMO_DIAR_BENCH` (load-time tick-cost bench + fused-vs-legacy self-check).

## Reproduce
- Identity / mixed: `rsi-bench/nemotron_diar/xs_correctness.py identity|mixed --target MID/DID`.
- Full set: `STREAM_MID=.. STREAM_DID=.. stream_bench_q.py --name xs2_notsofar --dirs data/notsofar_eval/audio:data/notsofar_eval/rttm --latency low --concurrency 6`.
- Ladder: `LG_MODEL/LG_DEP=<k6 runner> LG_SCRIPT=loadgen/diar_stream_k6.js loadgen/lg_arms.py <out> MID DID 250,320,400,480 --audio-s 3600 --control MID/DID --chunk-s 1.04 LATENCY=low`.
- Per-tick budget: deploy with `NEMO_DIAR_PROF=1 NEMO_DIAR_BENCH=1`; the session-end log line carries
  `handler={...}` (per-frame / per-partial µs, loop busy, process cores) and `stepper={...}` (phases, B
  histogram, GPU ms per row from CUDA events).
