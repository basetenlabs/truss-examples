# Nemotron 3 Diarized Transcription (batch) — cpWER benchmark

Measured on Baseten **fde-internal**, 1× **RTX-PRO-6000** (shipped config: shape-pinned, ASR
fp32, diarizer bf16 — see *Numerical determinism*). Metric: **cpWER**
(meeteval, via `stt_benchmark.diarization.scoring`) — the speaker-attributed WER we use for
all T+D comparisons. Dataset: **NOTSOFAR-1 eval_full_with_GT, first 30 sessions** (sorted
ids MTG_32000…MTG_32053), single distant channel, refs built from the ground-truth
`gt_transcription.json` (word-level timings + text + speaker).


## General Access diarizer (2026-09-19) — current headline

The sections below were measured with the early-access **preview** diarizer and are kept as history.
With the GA diarizer checkpoint (retrained bf16, same profiles; the ASR unchanged) and no code change,
eval-30 closed-loop K=30: **cpWER 28.58 macro / 28.89 micro** (preview 30.17 / 30.46; paired −1.59, GA
better on 22/30; ins 1234 vs 1362, del 7499 vs 8016, sub 3956 vs 3998); compute median 12.8 s per session
(15.3 s); speaker count mode 5, same as preview on 21/30. `rsi-bench/nemotron_diar/release/GA_VALIDATION.md` §6.

## Results — MATCHED (NOTSOFAR eval-30, identical sessions, refs and scorer)
| system | N | cpWER macro | micro | ins / del / sub | spk-count exact |
|---|---|---|---|---|---|
| **multitalker-Parakeet + Nemotron-3-Diarization** — *this truss, shipped config* (shape-pinned, ASR fp32, diarizer bf16, ≤8 spk) | 30 | **31.90** | **32.23** | — | — |
| same pipeline, first measurement (bf16, unpinned B=1 shapes — one bf16 shape draw, see *Numerical determinism*) | 30 | 31.13 | 31.66 | 2148 / 7886 / 3869 | 14/30 |
| ours: Qwen3-ASR-1.7B + **Nemotron-3 offline turns** (`../../nemotron-3-diarization-preview/batch`) | 29* | **31.78** | **32.25** | 1293 / 6920 / 5280 | 16/29 |
| ours: Qwen3-ASR + oracle turns (same packing) | 30 | 34.74 | 35.29 | 3059 / 4292 / 8146 | 30/30 |
| ours: Qwen3-ASR per reference segment (oracle-seg) | 30 | 34.80 | 35.38 | 3759 / 2849 / 8931 | 30/30 |
| ours: Qwen3-ASR + pyannote precision-2 turns | 30 | 44.31 | 45.05 | 3985 / 9505 / 6295 | 15/30 |
| ours: Qwen3-ASR + pyannote community-1 turns | 30 | 45.06 | 45.65 | 2879 / 10834 / 6337 | 17/30 |
| multitalker + Sortformer-4spk-v2.1 (fp32, ≤4 spk) — *not shipped* | 30 | 47.20† | 47.58† | — | always 4 (cap) |

\* Qwen3-ASR's `/v1/audio/transcriptions` falls into deterministic generation loops on some
packed chunks; MTG_32008 never completed at 28/20/15/10 s packing. On the 29 sessions every arm
scored: multitalker **31.01 / 31.50** vs ours+Nemotron 31.78 / 32.25 — multitalker better on
20/29 sessions, mean +0.77 pt; the whole edge is on 4-speaker sessions (23.3 vs 27.1, n=7),
dead even at 5 speakers (33.5 vs 33.3, n=22). † scored under the naive normaliser (~2 pt high).

**Full eval (129 sessions)**: first (bf16, unpinned) config **28.70 macro / 30.07 micro**, speaker
count exact 72/129; **shipped config at K=8: 28.94 macro / 30.26 micro** (`fixb_k8_full129`,
808 s wall for 129 sessions). The eval-30 is a slightly hard subset (the other 99 score
27.97 / 29.59 under the first config).

Superseded indicative numbers (kept for the record): our best dev-18 stack Qwen3-ASR + DiariZen
35.94 / 37.39, oracle 41.18, AssemblyAI 45.34 — different sessions, so not comparable to the above;
the first-pass multitalker figure 33.03 / 33.47 was a scorer artifact (see note).

Per-session cpWER for this truss ranges 10.4–59.6 (30/30 sessions improved vs the earlier
score by a uniform 0.4–3.8 pt; hypotheses are byte-identical).

**Scorer note.** The 33.03 figure came from an environment without `whisper_normalizer`, where
`stt_benchmark`'s `normalize_text` silently falls back to a naive lowercase/de-punct normaliser.
The canonical scorer (stt-benchmark venv: Whisper `EnglishTextNormalizer`, meeteval 0.4.3) gives
31.13 / 31.66 on the *same* hypotheses. Rows marked * were scored under the naive normaliser and
would shift by a similar ~2 pt; the dev-18 baselines were scored canonically.

## Read
- **A tie, not a win.** Under one scorer on the same sessions the coupled NVIDIA pipeline beats
  our best available stack (Qwen3-ASR + Nemotron-3 turns) by <1 pt — inside per-session noise,
  and only on 4-speaker meetings. The earlier "5-pt lead" was the normaliser artifact plus a
  dev-18 vs eval-30 mismatch.
- **The diarizer dominates, not the coupling.** Qwen + pyannote (precision-2 / community-1) turns
  is ~13 pt worse than Qwen + Nemotron turns on this far-field single-distant-channel audio:
  fragmented or missed far-field turns cost the ASR words, not just attribution.
- **cpWER is ASR-dominated and Nemotron's turns help the ASR.** Oracle diarization (34.7) scores
  *worse* than Nemotron's own turns (31.8) under identical packing — long merged turns give the
  ASR more context than tight word-level reference segments.
- What the coupled pipeline uniquely buys is transcription of fully **overlapped** speech and a
  single-model streaming path; on aggregate cpWER it is not a reason by itself to switch.
- The **Sortformer-v2.1 pairing** tracks Nemotron on 4-speaker sessions (25.6 vs 25.8,
  21.9 vs 21.0) and collapses on 5–7-speaker ones (73–77 vs 30–60): the gap is entirely the
  4-speaker cap.
- **DER from this system is not meaningful** (161–163 % at 0.25): it emits transcript-level
  sentence spans (30 s sentence-break merging), not frame-level speaker activity.

## Numerical determinism (why the headline moved from 31.13 to 31.90)
NeMo's lockstep step runs the ASR encoder once over ΣS = (sessions × active speakers) rows in bf16
autocast. On RTX-PRO-6000, cuBLAS picks split-K / stream-K bf16 GEMM kernels for some (M, N, K)
whose reduction order depends on the output tile: **two identical rows in one call differ by one
bf16 ulp** (first non-invariant op `encoder.layers[1].self_attn.linear_q` at M = 224), and the
kernel — hence the rounding — also changes with M. The 24-layer streaming conformer amplifies one
ulp (5e-4) to ~0.5 at the encoder output within a step, the caches carry it forward, and greedy RNNT
plus speaker attribution turn it into different words. Consequences measured on the eval-30:

- 8 identical copies of one file in one B=8 batch → 8 distinct transcripts, none equal to B=1.
- Per-session cpWER between two equally valid shape draws has **SD 5.7** (MTG_32049 +23.5,
  MTG_32027 −15.8 between unpinned B=1 and B=8). Any aggregate is a draw: 31.13 (first bf16 B=1
  config), 31.89 (unpinned micro-batched K=8), 31.91 (bf16 pinned), 31.90 (ASR fp32 pinned) are
  all within 0.8 macro, i.e. < 1 SE of the per-session noise. **The previously quoted 31.13 was
  one bf16 shape draw, not a better system.**
- Not caused by the audio zero-padding (worst batches had 1 % padding) and not a NeMo indexing
  bug — every input to the diverging call was bit-identical across rows (row tracer, see
  `rsi-bench/nemotron_diar/results_mt_td/B_GT_1_ROOTCAUSE.md`).

Fix shipped in `packages/multitalker_batch_fix.py`: **pin the shapes** — zero-pad every ASR call
to `MT_ASR_ROWS` (= cap × max_speakers = 64) speaker rows and every diarizer call to
`MT_DIAR_ROWS` (= cap = 8) sessions, slice the results back — so B = 1…cap and any ΣS execute the
same kernels; and run the ASR step in **fp32 without autocast** (`MT_ASR_FP32=1`), which is
row- and shape-invariant at every shape. Three configs, eval-30 at K=1, each run twice:

| config | cpWER macro | micro | run-to-run identical | compute / 6-min session (median) | peak GPU |
|---|---|---|---|---|---|
| (a) bf16 ASR + bf16 diarizer, pinned | 31.91 | 32.27 | 30/30 | 17.4 s | 4.63 GB |
| **(b) fp32 ASR (no autocast) + bf16 diarizer, pinned — shipped at e98c7e8** (now: bf16 ASR pinned 32, turn-builder output, 30.17) | **31.90** | **32.23** | **30/30** | **21.6 s** | 4.63 GB |
| (c) fp32 ASR + fp32 diarizer, pinned | 33.24 | 33.56 | 30/30 | 27.8 s | 4.86 GB |
| unpinned bf16 B=1 (first measurement, `prod_k1`) | 31.13 | 31.66 | 30/30 | 13.0 s | 3.7 GB |

(b) is chosen: same cpWER as (a), deterministic, ASR invariant to any shape (also what NeMo's
CUDA-graph encoder wrapper requires for the streaming preset), for +25 % compute at B=1 — the
pinned fp32 encoder runs 64 rows every step whether one or eight sessions are present, so the
extra cost is flat, not per session. (c) is worse on cpWER and +60 %. Residual: the bf16 diarizer
is still row-*position* dependent at a few sync-mode shapes (~1e-4 in speaker posteriors), so a
session alone vs the same session as row b of a batch can differ when that crosses the 0.5
activity threshold — measured below as the K=8 vs K=1 identity count.

## Output path — turn builder instead of NeMo's seglst: cpWER 31.90 → **30.17** (2026-09-17)

Same audio through the streaming preset scored 29.48 with the same scorer (cross-checked: batch hypotheses through the
streaming harness's scorer = 31.899). The decomposition on eval-30 located the gap in **insertions** — batch 2,192
(fp32) / 2,183 (bf16) / 2,148 (unpinned B=1 with a 1-row diarizer) vs streaming 1,322, deletions equal (7,917 vs
7,946), substitutions +200; the worst files are insertion bursts only in batch (MTG_32048 14 → 309, MTG_32004
34 → 197, MTG_32026 43 → 178). Mechanism: `ASRState.update_sessionwise_seglsts_for_parallel` appends
`text.strip()` — the speaker's whole transcript so far — whenever the new hypothesis text is not a string-prefix
extension of the previous one ("non-prefix hypothesis revision" fallback), which a punctuation piece triggers by
removing the space before it. The streaming preset never sees it because `packages/mt_turns.py` rebuilds words
from the append-only token ids and RNNT emission frames. The batch preset now emits the same turn builder's
segments (`MT_TURN_SEGMENTS=1`; 0 = NeMo's seglst for A/B):

| output path (cap 16, bf16 pinned 32, 8-row diarizer slabs, K=30) | cpWER macro / micro | ins / del / sub | files improved |
|---|---|---|---|
| NeMo seglst (`MT_TURN_SEGMENTS=0`) | 31.90 / 32.23 | 2,192 / 7,917 / 4,044 | — |
| **turn builder (shipped)** | **30.17 / 30.46** | 1,362 / 8,016 / 3,998 | 22/30, mean −1.73, best −16.3 (MTG_32048), worst +1.8 |
| streaming preset, mode 0, same audio | 29.48 | 1,322 / 7,946 / 3,851 | |

Compute is unchanged (post-processing; 15.3 s per 16-session batch). The residual +0.69 vs streaming tracks the
diarizer's row count (streaming runs it per session, 1 row), but it is a draw, not a monotonic bias — measured with
the turn-builder output, cap 16, K=30:

| `MT_DIAR_ROWS` (diarizer slab) | cpWER macro / micro | ins / del / sub | compute per 16-session batch |
|---|---|---|---|
| **8 (shipped)** | **30.17 / 30.46** | 1,362 / 8,016 / 3,998 | **15.3 s** |
| 4 | 30.41 / 30.69 | 1,427 / 8,231 / 3,821 | 17.9 s (+17 %) |
| 2 | 30.56 / 30.82 | 1,471 / 8,280 / 3,784 | 21.5 s (+40 %) |
| streaming preset, 1 row per session | 29.48 | 1,322 / 7,946 / 3,851 | (16 replays per step here: ~+60 ms on a ~45 ms step) |

Fewer rows per slab cost compute and did not move cpWER in our favour, so 8 stays.

## Capacity from the cloud (2026-09-17) — cap 16, bf16 pinned, diarizer slabs: 61 sessions/min closed-loop, 50/min open-loop, cpWER 31.90 at load

All rows below were driven from an in-cluster CPU truss (`rsi-bench/nemotron_diar/loadgen/mt-batch-runner`,
model `nemotron-mt-td-batch-loadgen`; `mt_batch_lg.py`), never from a laptop; every run's last response per
file is cpWER-scored with the canonical scorer, so throughput and quality come from the same run. Closed =
K workers looping over the eval-30 files for 300 s; open = one request every 60/rate s for 600 s (the
sustainable ceiling is the highest rate whose latency is flat across the run's thirds).

| build (one RTX-PRO-6000) | arm | sessions/min | latency p50 (by third) | B | compute | cpWER at load |
|---|---|---|---|---|---|---|
| cap 8, fp32 ASR, `no_while_loops` | closed K=16 | 31.6 | 30.2 s (30.3 / 30.0 / 29.6) | 8 | 13.9 s | 31.899 / 32.225 |
| | open 30/min | 28.8 (holds) | 23.4 s (23.7 / 23.6 / 23.4) | 6–8 | 13.6 s | 31.908 / 32.234 |
| | open 40/min | **fails** | 41 → 83 → 123 s | 8 | 13.8 s | 31.899 |
| cap 16, fp32 ASR, 8-row diarizer slabs | closed K=16 | 27.8 | 33.8 s (flat) | 15 | 21.6 s | 31.894 / 32.221 |
| | closed K=30 | 41.7 | 41.9 s (flat) | 14–16 | 20.7 s | 31.899 / 32.225 |
| | open 30/min | 29.2 (holds) | 23.1 s (22.6 / 23.4 / 23.4) | 6–9 | 13.5 s | 31.908 |
| | open 40/min | 37.9 (holds) | 28.3 s (27.9 / 28.3 / 28.9) | 11–14 | 17.7 s | 31.899 |
| | open 50/min | **fails** | 47 → 84 → 121 s | 16 | 23.0 s | 31.899 |
| **cap 16, bf16 ASR (pinned 32), 8-row diarizer slabs — shipped** | closed K=30 | **61.0** | 28.8 s (28.9 / 28.7 / 28.7) | 14–16 | **14.5 s** | **31.899 / 32.214** |
| | open 50/min | **49.3 (holds)** | 13.8 s (13.9 / 13.6 / 13.9) | 6–8 | 8.0 s | 31.899 |
| | open 70/min | 65.0 (**not sustained**: latency rising) | 26 → 37 → 47 s | 16 | 14.7 s | 31.899 |

Read: cap 8 sustains ~35/min, fp32 cap 16 ~42/min, bf16 cap 16 ≥ 50/min sustained (70/min served at 65/min with a
slowly growing queue) and 61/min closed-loop at K=30 — 3× the ≥ 21 bar. bf16 vs fp32 per file: mean |Δ| 0.38, max 1.77, macro 31.908 vs 31.899 (K=8 laptop
correctness runs, same bytes at every K because the rows are pinned). bf16 is **not** shape-invariant — M=8 vs
M=32 pinned rows: 0/30 files byte-identical, mean |Δ| 0.26 / max 1.76, with or without cuBLAS's reduced-
precision split-K reductions (at M=32 the two settings are byte-identical, 30/30; at M=8 they differ on 22/30
files, mean |Δ| 0.17) — so the shape dependence is not the split-K accumulation, and the rows stay pinned. fp32
(`MT_ASR_DTYPE=fp32`) remains the shape-invariant option (M=8 == M=32 byte-identical, 29/30 files identical
between K=16 and K=30). The RNNT decoder runs `no_while_loops` (identical bytes and speed to `full_graph`,
which faulted intermittently — see the `MT_FAST` section). Validation deployments (`nemotron-mt-td-batch-
fast8/-fast16/-fast16s/-f16bf16`, `bf16-p8/-p32`(`-def`), `dbg64`), all deactivated.

## Step-time levers (`MT_FAST=1`) — first cloud-independent measurement: 15.9 → 25.1 sessions/min at K=16, cpWER unchanged (history; the shipped cap-16 bf16 build is in "Capacity from the cloud" above)

NeMo's lockstep step is launch-bound (~3,000 kernel launches per step). `packages/mt_fast.py` — the
streaming preset's performance build, shared file — dispatches the same NeMo calls through CUDA graphs:
the sync-mode diarizer core replayed from one graph per sequence length (bit-identical to eager; 134
graphs captured by a 180 s warm session at load), NeMo's `CudaGraphsStreamingEncoderStep` for the
fp32 encoder, shallow hypothesis copies, incremental detokenisation. The ASR runs in **32-row slabs**
(`MT_PAD_ROWS`): the first graphed 64-row call faults on this NeMo build (`cudaErrorIllegalAddress`
inside the label-looping decoder's `full_graph.replay()` at batch 64, fp32 and bf16 alike, pinned
with `CUDA_LAUNCH_BLOCKING=1`; 8/16/32 never), and fp32 is shape-invariant, so 2 × 32 == 1 × 64.
Eval-30, canonical scorer, closed-loop client (`mt_td_bench_prod.py`), one RTX-PRO-6000:

| build | K | cpWER macro / micro | total wall (30 sessions) | sessions/min | batch compute (B=8) | peak GPU |
|---|---|---|---|---|---|---|
| e98c7e8 eager, bf16 ASR pinned 64 rows (`audit_bf16_k16`) | 16 | 31.91 / 32.26 | 113.4 s | 15.9 | 20.0–21.7 s | 5.2 GB |
| **`MT_FAST=1`, fp32 ASR, 32-row slabs, cap 8 (`fast8_k16`)** | 16 | **31.91 / 32.23** | **71.7 s** | **25.1** | **12.3–13.7 s** | 6.6 GB |
| same, all 30 at once (`fast8_k30`) | 30 | 31.90 / 32.23 | 65.1 s | 27.6 | 13.2–16.0 s (B=7–8) | 6.6 GB |
| cap 16 (`MT_MB_CAP` 16, `MT_DIAR_ROWS` 16; `fast16_k30`) — rejected | 30 | 32.46 / 33.01 (**+0.56**) | 58.4 s | 30.8 | 20.5 s (B=16) | 8.5 GB |

The fp32 fast build reproduces the shipped numerics: 31.90 / 32.23 = config (b) below to the digit,
29/30 files byte-identical between K=16 and K=30 (the 30th is the bf16 diarizer's row-position
residual noted below). Steady state at cap 8 is 8 sessions per ~13.7 s ≈ **35 sessions/min per
replica** when the batcher stays full; the closed-loop figures include the client's FLAC encode/upload
and the partial last batch. Cap 16 clears 30 sessions/min but fails the quality gate: **the +0.56 is the
diarizer padded to 16 rows** (`MT_DIAR_ROWS`), not the ASR — the earlier bf16-ASR cap-16 run
(`audit_cap16_k24`) scored the same 32.50, per-file deltas reach ±22. Running the diarizer as 8-row
slabs inside a 16-session batch would keep the cap-8 draw at ~47 sessions/min (B=16 compute 20.5 s);
not done here.

## Cost (RTX-PRO-6000, e98c7e8 eager build; the shipped `MT_FAST=1` build is ~1.6× faster, above)
Median server compute **21.6 s per ~6-min session at B=1 (RTFx ≈ 17×)**, wall median 23.4 s
(FLAC upload + decode); the first bf16 unpinned config was 13.0 s — the pinned fp32 encoder
(64 rows every step) costs ~+4 s per session flat. Peak GPU **4.6 GB** at B=1, **5.1–5.3 GB** at
B=7–8. Load-time warmup (one B=1 pass; shapes are pinned so it warms every batch size) makes the
first request fast.

### Request micro-batching (history: e98c7e8 eager build, `predict_concurrency: 16`, cap 8)
Requests arriving within `MT_MB_WINDOW_MS` (100 ms) — plus everything queued while the previous
session ran — form ONE coupled session with B files on NeMo's batch dimension
(`instance_manager.batch_asr_states[b]`), rows zero-padded to the longest file and cut back at
their true duration. The step is launch-bound (~50 ms/step for ~11 ms of GPU work), so B rows cost
about the same wall as one until the GPU fills. Distinct ~6-min sessions, closed-loop client at
concurrency K:

| K | sessions | total wall | per session | sessions/min/replica | batch compute (B) | GPU util |
|---|---|---|---|---|---|---|
| 1 | 30 | 729.7 s | 24.3 s | 2.5 | 21.6 s (B=1) | 68–71 % |
| 4 | 8 | 105.0 s | 13.1 s | 4.6 | 25.4 s (B=2–4) | 72–74 % |
| 8 | 8 / 30 / 129 | 54.3 / 198.5 / 808.2 s | 6.8 / 6.6 / 6.3 s | 8.8 / 9.1 / 9.6 | 23.6 s (B=7) | 68–69 % |
| 16 | 16 | 83.9 s | 5.2 s | 11.4 | 29.2 s (B=8, 437 s longest) | 67 % |

GPU util is flat at ~70 % from B=1 to B=8 (NVML sampler): the pinned fp32 encoder already runs
the 64-row step at B=1, so the extra sessions are literally free until the cap. A B=8 session
costs 1.1–1.35× a B=1 session, so steady state at cap 8 is ~16–19 sessions/min/replica (6–7×
unbatched); the client
numbers above include the closed-loop harness forming 7+1 batches (one client is always out of
phase). Cap 16 (`MT_ASR_ROWS` 128, `MT_DIAR_ROWS` 16) was measured and rejected: 128 fp32 rows
make the step GPU-bound (85 % util at B=1, 34–36 s alone; 38–48 s per 13–15-row batch, 1.8–2.2×
the cap-8 B=1) — worse for everyone below B≈12.

Parity: K=8 vs K=1 (same shipped config) **28/30 sessions byte-identical**, macro 31.91 vs 31.90;
the two others differ by one segment boundary (MTG_32007 +0.15, MTG_32049 ±0.00 cpWER) — the
diarizer's bf16 row-position residual noted above. Stability: 13 consecutive sessions (7 of them
full B=8, 88 requests pipelined two rounds deep) with no error; the replica served >150 sessions at
B=1–8 over its lifetime with no CUDA fault. (The unpinned micro-batcher had died twice with
`CUDA error: an illegal memory access` after mixed-B sessions — most plausibly the RNNT
label-looping decoder re-capturing its CUDA graph mid-session when ΣS grew; pinning sizes the
decoder once at 64 rows, so the re-capture never happens.)

Capacity ≈ replicas × ~16 sessions/min at K≥8; per-request latency ≈ one session
(~22–25 s) plus queueing behind the in-flight session. Requests with different `max_speakers`
never share a session (it binds per session); `MT_MB_PAD_WASTE` (0.3) splits a window when
padding to the longest file would waste more than that fraction of rows (e.g. a 60-min file
arriving with 1-min files). Duration grouping never triggered on the eval-30 (pad ≤ 15 %).

## Reproduce
`rsi-bench/nemotron_diar/mt_td_bench_prod.py` (env `NEMO_MID`/`NEMO_DID`, run with the
stt-benchmark venv) over `rsi-bench/nemotron_diar/data/notsofar_eval/{audio,seglst}`;
`mt_td_compare.py --tag X --base Y` gives per-session deltas and byte-identity; `mt_td_rowtest.py`
fires identical rows into one batch; `mt_td_b8loop.py` runs consecutive tight B=8 sessions.
Per-session hyps + `summary.json` under `rsi-bench/nemotron_diar/results_mt_td/`: `prod_k1`
(first config), `fix{a,b,c}_k1_r{1,2}`, `fixb_k8`, `fixb_thr_k{4,8,16}`, `fixb_k8_full129`.
