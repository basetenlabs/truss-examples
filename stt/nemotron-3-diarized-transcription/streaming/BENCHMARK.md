# Nemotron 3 Diarized Transcription (streaming) — benchmark

Measured 2026-09-15/16 (PT) on Baseten **fde-internal**, 1× **RTX-PRO-6000** per replica, from the
experiment copies this preset is a port of (`rsi-bench/nemotron_diar/mt-td-stream-truss` = eager build,
`mt-td-stream-truss-v2` = this performance build). Clients: `rsi-bench/nemotron_diar/mt_fast_client.py`
(budget / cpwer / load / long), `mt_stream_client.py`, `mt_load_test.py`. Full write-up with raw data:
`rsi-bench/nemotron_diar/results_mt_stream/FASTSTEP.md` (+ `PROFILE.md`, `NEMO_MULTITALKER_ARCH.md`).

**Indicative, not a full eval.** Correctness is on **3 files**; the batch preset's 30-session figure
(`../batch/BENCHMARK.md`) is the quality number — this document shows what streaming reproduces and what
the performance build costs and buys.


## General Access diarizer (2026-09-19) — what changed

Everything below was measured with the early-access **preview** diarizer and is kept as history. With the
GA diarizer checkpoint and no code change: 3-file FINAL cpWER 30.58 / 22.74 / 15.78 (macro 23.03 vs 23.47),
eval-30 batch cpWER 28.58 (`../batch/BENCHMARK.md`); 190-stream hold and latency not re-run on GA. Turn
defaults changed 2026-09-22 (sentence-break off, pause 2.5 s, hand-over 2.0 s / 6 words, freeze closed):
formatting only, same words. `rsi-bench/nemotron_diar/release/GA_VALIDATION.md` §6,
`rsi-bench/nemotron_diar/results_aai_killer/WORKSTREAM_A.md`.

## Correctness — FINAL cpWER (canonical scorer, whole NOTSOFAR-1 eval sessions)

| build | MTG_32000 | MTG_32003 | MTG_32004 | macro | deterministic |
|---|---|---|---|---|---|
| eager build (= `../batch`; fresh replica reproduces RESULTS.md to the digit) | 23.99 | 28.31 | 19.12 | 23.81 | yes (B=1 shapes fixed) |
| **this build** (bf16 weights, padded rows) | 23.57 | 28.38 | 18.45 | **23.47 (−0.34)** | **yes** — two passes byte-identical |
| `MT_ASR_DTYPE=fp32` option | 24.20 | 28.45 | 18.52 | 23.72 (−0.09) | yes |

The −0.34 is not a quality change: every GEMM's bf16 rounding depends on its batch shape and the 24-layer
conformer + greedy RNNT amplify one ulp into different words (`rsi-bench/nemotron_diar/results_mt_td/
B_GT_1_ROOTCAUSE.md`: per-session SD between two equally valid shape sets is 5.7 cpWER, macro band ±0.8).
The eager build re-rolls those shapes every time the active-speaker count changes; this build pins them
(8 padded rows) so the transcript is a single, reproducible draw. Each lever's own effect on the output was
checked separately (ladder below): mel, no-deepcopy and the diarizer graphs are output-identical, the
encoder graphs are bit-identical in fp32, bf16 weights are the −0.34.

## Per-step budget — before / after (N=1, MTG_32000, k = 1–2 active speakers, median ms, lock-held)

| phase | eager build | this build | how |
|---|---|---|---|
| mel (outside the lock) | 12.0 | 1.0 | one preprocessor call per 112-frame step from a bounded ring (was ~11 calls/step on a growing buffer) |
| diarizer step | 21.5 | 3.8 | sync-mode core replayed from a per-length CUDA graph (bit-identical to eager) |
| gate + pre-encode | 1.6 | 0.2 | pre-encoder runs inside the encoder graph |
| gather | 0.9 | 0.26 | no per-step deepcopy of hypotheses |
| ASR encoder (FastConformer, cache-aware) | 26.7 | 5.0 | NeMo `CudaGraphsStreamingEncoderStep`, bf16 weights, one graph at 8 padded rows |
| RNNT decoder (label-looping, `full_graph`) | 2.5 | 3.0 | unchanged (+0.5 from 8 rows) |
| **step (lock held)** | **53.4** | **12.7–13.0** | **4.1×**; ~3,100 kernel launches → a handful of graph replays |

## Lever ladder (each row = the previous plus one lever; cpWER on the 3 files)

| levers | step ms | diar | enc | macro cpWER | Δ vs 23.81 |
|---|---|---|---|---|---|
| eager build | 53.4 | 21.5 | 26.7 | 23.81 | 0 |
| bf16 weights | 48.2 | 21.6 | 21.5 | 23.47 | −0.34 |
| + mel once per step | 49.0 | 21.7 | 21.5 | 23.47 | identical ins/del/sub |
| + encoder CUDA graphs | 31.0 | 21.8 | 4.8 | 23.53 | −0.28 |
| + diarizer sync-mode CUDA graphs | 12.1 | 4.0 | 4.8 | 23.53 | identical to the row above (bit-exact) |
| + no deepcopy | 12.1 | 4.0 | 4.8 | 23.53 | identical |
| **+ 8 padded rows (shipped)** | **13.0** | 3.8 | 5.0 | **23.47** | −0.34, deterministic |
| rejected: async (fixed-shape) diarizer state | 31.1 | 21.9 | 4.8 | 24.87 | **+1.06** |
| rejected: + `torch.compile` of the diarizer core | 13.4 | 4.4 | 4.8 | 28.82 | **+5.0** |
| option: fp32 weights + graphs + padded rows | 20.0 | 3.9 | 11.7 | 23.72 | −0.09 |
| rejected: fp32 + TF32 (encoder-scoped) | 14.1 | 3.8 | 6.1 | 22.99 | −0.82 (outside the band) |

Rejected levers: the async diarizer follows the *same* cache/FIFO schedule as sync mode (checked per step
with a shadow state) but its bf16 attention over a 542-frame padded sequence rounds differently from step 0
(1e-2 in the posteriors), the compression then picks different frames, and a flipped 0.5 speaker-activity
decision changes which ASR instances run — MTG_32004 insertions 33 → 107, 6 hypothesised speakers.
`torch.compile` fuses kernels and adds a second rounding change. Both remain in the code as `MT_DIAR_ASYNC`
/ `MT_DIAR_COMPILE` for A/B only.

## Concurrency curve (real time, 90 s of MTG_32000, 100 ms PCM16 chunks)

`lag` = wall-clock behind live audio when a partial covering that instant arrives; `backlog` = late-quarter
p50 − early-quarter p50 (grows when the replica cannot keep up).

| N | eager build p50 / p95 / backlog | this build p50 / p95 / backlog | lock-held step (this build) | GPU util mean |
|---|---|---|---|---|
| 15 | 0.54 / 1.06 / +0.13 | 0.16 / 0.28 / +0.03 | 12.5 ms | 13% |
| 20 | 6.15 / 13.21 / +9.95 (falling behind) | 0.20 / 0.35 / +0.04 | 13.3 ms | 16% |
| 30 | collapse | 0.27 / 0.47 / +0.02 | 12.4 ms | 25% |
| 40 | 32/40 sessions, collapse | 0.34 / 0.62 / +0.10 | 13.5 ms | 34% |
| **60** | — | **0.54 / 0.91 / +0.09** | 13.5 ms | 51% (median 67%) |
| 80 | — | 1.27 / 2.61 / +1.36 (falling behind) | 13.5 ms | 64% |

Lock-path ceiling (history) **60 streams per replica** (the eager build's N=10 latency); 80 is over the edge (80 × 13.5 ms =
96% lock occupancy — also near the laptop client's uplink limit). The step does not inflate with N
(11.5–13.5 ms at every level; the eager build went 53 → 76 ms at N=15 from GIL / default-stream
interference by the per-append mel, which the per-step mel removes). History: this lock-path ceiling set
`predict_concurrency` 64 at the time; the shipped stepper build holds 190 hour-long and 220 in a 20-min screen
(`predict_concurrency: 220`, "2026-09-17 build" below).

## Long audio (60-minute sessions) — `rsi-bench/nemotron_diar/results_mt_stream/LONGAUDIO.md`

Audio: the first ten NOTSOFAR-1 eval sessions concatenated and cut at 3600 s (4–6 speakers each, ~40 label
changes over the hour). One stream, non-real-time, synchronize-bounded phase timers, medians (ms):

| minute | 0 | 10 | 20 | 30 | 40 | 50 | 59 | first 5 → last 5 min |
|---|---|---|---|---|---|---|---|---|
| NeMo detokenisation: step / dec | 13.2 / 2.8 | 15.1 / 4.1 | 14.8 / 4.0 | 15.3 / 4.3 | 17.2 / 6.0 | 16.5 / 5.2 | 15.8 / 5.0 | 13.9 → 15.9 / 3.0 → 5.0 |
| **`MT_INC_DETOK` (shipped): step / dec** | 13.4 / 2.8 | 13.9 / 2.8 | 13.8 / 2.9 | 13.6 / 2.7 | 14.2 / 3.0 | 13.7 / 2.7 | 13.5 / 2.7 | **13.6 → 13.6 / 2.8 → 2.7** |
| diar / enc / residual (both) | 3.5 / 5.0 / 0.95 | flat | flat | flat | flat | flat | 3.8 / 5.0 / 0.95 | |

The growth was `Hypothesis.merge_` (`rnnt_utils.py:186`) dropping `text` so `decode_hypothesis`
(`rnnt_decoding.py:813-836`) re-ran `tolist` → `id_to_piece` → SentencePiece `decode_pieces` → the punctuation regex
over the whole session per speaker per step. `packages/mt_incremental.py` caches the committed prefix on the
hypothesis (cut at word-start pieces, where SentencePiece decoding and the `(\s)(punct)` regex are compositional) and
decodes only the tail. Identity: 3-file FINAL cpWER unchanged (23.57 / 28.38 / 18.45, seglst JSON byte-identical),
60-min FINAL seglst byte-identical (164 segments / 15,623 words; also identical across two replicas), and with
`MT_INC_DETOK_VERIFY=1` 25,056 per-step decodes over the hour matched NeMo's full decode with 0 mismatches.
`seglst` still grows 0.15 → 0.22 ms over the hour (NeMo re-sorts the session's sentence list each step) — left alone.

### Hour-long streams per replica (k6 in-cluster, real time, 60 min of audio per stream)

Generator: a k6 v0.54 CPU-only truss in fde-internal (`rsi-bench/nemotron_diar/loadgen/`), one VU per 60-minute session,
100 ms PCM16 frames paced to the wall clock, ramp 30 s, a concurrent 1-VU control against an idle second replica; nothing was
driven from a laptop. Emit lag = wall time a partial arrives minus the wall time the frame that completed its `processed_s`
left the generator. Ladder 20 → 40 → 60 → 80 (stopped at the first collapsed arm), then 64. Raw per-frame logs, k6
summaries, generator CPU and `MANIFEST.json` per arm: `results_mt_stream/sweep_2026-09-16/` (`SWEEP.md`).

| streams | done | emit lag p50 / p95 / p99 | worst 10-min window p95 | backlog excess at the end | final latency | server step p50 / p99 (steps > 100 ms) | lock occupancy | GPU util | idle-replica control p50 / p95 |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 20/20 | 0.21 / 0.33 / 0.35 s | 0.34 s | 0 | 0.17 s | 12.2 / 15.1 ms (0 of 64k) | 22 % | 17 % | 0.22 / 0.32 s |
| 40 | 36/40 (4 sockets reset by the gateway at min 15; no server error) | 0.22 / 0.34 / 0.36 s | 0.36 s | 0 | 0.18 s | 12.8 / 16.4 ms (0 of 119k) | 42 % | 32 % | 0.22 / 0.32 s |
| **60** | 60/60 | 0.22 / 0.34 / 0.36 s | 0.35 s | 0 | 0.18 s | 12.9 / 17.7 ms (1 of 193k) | 69 % | 51 % | 0.22 / 0.32 s |
| **64** | 58/64 (6 sockets reset by the gateway, the control's too; no server error) | 0.23 / 0.36 / 0.53 s | 0.53 s (connect burst during the ramp; 0.30–0.38 s after) | 0 | 0.17 s | 13.2 / 16.6 ms (0 of 203k) | 74 % | 54 % | 0.22 / 0.33 s |
| 80 | 80/80, all behind | 2.2 / 10.6 / 14.9 s | 12.1 s | **7.0 s** | 4.5 s | 13.8 / 17.3 ms (2 of 257k) | **97 %** | 69 % | 1.0 / 8.2 s (ingress path congested too) |

The step itself never inflates (12–14 ms p50, p99 ≤ 18 ms, no periodic stalls: at most 2 steps > 100 ms in a quarter million);
80 fails because 80 × 13.8 ms is 99 % of the 1.12 s window — the lock queue (wait p50 750 ms) becomes the latency. The lag
staircase inside each hour (0.09 → 0.15 → 0.22 s at ~10 and ~30 min) is the replace-style partial's growing wire size
crossing TCP windows (≤16 / 16–32 / >32 KB; ~86 KB at the end of the hour) and is identical on the idle control; the server's
own contribution over the control is 0.00–0.03 s at 20–60. GPU memory scales with concurrency, not time (24 GB at 20, 29 GB at
60–80). Verdict: **60 hour-long streams per RTX-PRO-6000 hold for the full hour; 80 collapse** — **64 holds on the server** (lock 74 %, lock wait p99 92 ms, no backlog) with completion 58/64 capped by gateway socket resets that also took the idle-replica control, so **64 hour-long streams per replica** stood for the lock path (`predict_concurrency` 64 at the time; the lock path's ceiling is ~64–70). The shipped stepper build holds 190 hour-long and 220 in a 20-min screen — "2026-09-17 build" below.

### Hour-long streams per replica — cross-session batched stepper (`MT_XSESSION=1`, `MT_XDIAR=0`; `SWEEP_XSESSION.md`)

Same generator, scenario, audio, pacing, 30 s ramp and 1-VU idle control (my own instance of the k6 runner truss; the
scenario copy logs every frame in full), registry layout at `31bafd6` with `predict_concurrency: 220` / `MT_WORKERS: 300` /
`MT_PROFILE: 1` for the per-step records (the stepper never takes the `MT_PROFILE_EVERY` profiler snapshot). Ladder 60 → 80
→ 100 → 160 → 200. Tables from the sweep fork's `sweep_analyze.py` on the raw logs; the stepper cost is the deduplicated tick
(the analyzer sees a B-row tick B times). Raw logs and `MANIFEST.json` per level:
`results_mt_stream/sweep_2026-09-16_xsession/`.

| streams | done (gateway resets, no server error) | emit lag p50 / p95 / p99 | 10-min window p95 | backlog excess at the end | final latency p50 | tick p50 / p99 (@10/30/60 min p50) | stepper busy | B per tick p50 / p90 / max | GPU util | idle control p50 / p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| **60** | 57/60 (3) | 0.22 / 0.33 / 0.35 s | 0.21 → 0.34 s | 0 | 0.14 s | 12.9 / 24 ms (12.8 / 13.3 / 13.2) | 56 % | 1 / 2 / 16 | 43 % | 0.22 / 0.32 s |
| **80** | 65/80 (15, two clustered events at min 24 and 39) | 0.23 / 0.36 / 0.38 s | 0.24 → 0.37 s | 0 | 0.14 s | 17.3 / 51 ms (17.9 / 18.0 / 14.6) | 54 % | 2 / 6 / 16 | 40 % | 0.22 / 0.32 s |
| **100** | 78/100 (22) | 0.21 / 0.34 / 0.36 s | 0.21 → 0.35 s | 0 | 0.15 s | 14.7 / 32 ms (16.1 / 15.2 / 13.7) | 79 % | 1 / 3 / 16 | 58 % | 0.22 / 0.32 s |
| **160** | 120/160 (40; effective load 133 → 120 over the hour) | 0.25 / 0.48 / 0.69 s | 0.25 → **0.70** s (rising every window) | 0.32 s (p95) | 0.20 s | 25.4 / 65 ms (26.0 / 27.3 / 21.5) | **94 %** | 3 / 6 / 16 | 65 % | 0.22 / 0.32 s |
| 200 | 29/200 (171) | 55 / 1205 / 1331 s | 51 → 1336 s | **1265 s** | 44 s | 100 / 130 ms at the B=16 cap (98 / 109 / 112) | **98 %** | 16 / 16 / 16 | 64 % | 0.22 / 0.32 s |

Verdict at the time (superseded by the 2026-09-17 build: 190 hour-long, 220 screened): **160 hour-long streams per RTX-PRO-6000 hold the full hour (window p95 ≤ 0.70 s, no backlog) on that
`MT_XSESSION=1` / `MT_XDIAR=0` build; 200 collapse** (a 16-row tick with the per-row sync diarizer is ~100 ms ≈ 6 ms per
session-chunk; the stepper saturates while the GPU is at 64 %). `predict_concurrency` was set to 128, then 160, from this ladder (160 held at 94 % stepper busy
with p95 rising every window; 100 ran at 79 % flat) — 2.5× the lock path's 60–64.

The first hour-long N=60 run (`dc3cb0e`, before the sender mailbox) showed p95 5.5 s with an idle stepper: the sequential
`receive -> step -> await send_text` handler let a slow reader of the growing replace-style partials stall that connection's
audio path (inbound lag 0.7-1.3 s p50 on 43 of 60 connections). `31bafd6` fixed it; the same lag staircase with partial size
(0.21 → 0.34 s over the hour) as on the lock path and on the idle control remains. Gateway socket resets scale with the
connection count (3 / 15 / 22 / 40 of 60 / 80 / 100 / 160 per hour) and are counted separately from the server.

## 2026-09-17 build — connection-side levers, slot-table diarizer (`MT_XDIAR=1`), identity mel

Levers landed after the audit (commits 096fc8b, 0ee34ee): mel computed in the stepper from raw windows
(one STFT per tick, filterbank GEMM / normalisation per row so every row is bit-identical to the per-row
mel — checked on 165,034/165,034 rows at N=160), connections await an asyncio future instead of parking a
pool thread (`MT_WORKERS` 300 → 32), no executor hop for frames that do not complete a chunk, partials built
from cached per-turn JSON fragments (byte-identical to `json.dumps`), one torch CPU thread, and `MT_XDIAR=1`
rebuilt on preallocated per-slot state tables with NeMo's async update made sync-free and captured as one
CUDA graph per batch size (16 graphs in 5 s at load; nothing allocated per tick).

Quality gates (canonical scorer): `MT_XDIAR=0` reproduces the committed build to the digit — 3-file
23.57 / 28.38 / 18.45 with identical ins/del/sub, eval-30 29.48 = 29.48. `MT_XDIAR=1`: 3-file 23.64 / 28.02 /
18.18, eval-30 **29.27 (−0.21)**, per-file |Δ| mean 1.39 / max 11.3 (MTG_32009 −11.3, MTG_32026 +7.9) — the same
draws as the audit's XDIAR=1 pass, so the slot-table rebuild adds no deviation; inside the ±5.7 shape-draw
band, so it is the default now (bit-exact mode 0 stays a flag).

### Output under load (row buckets and tick batch vary with live sessions)
The ASR runs bf16 at pinned buckets of 8 / 16 / 32 rows and the `MT_XDIAR=1` diarizer at B = 1…16, both chosen
by the tick's load, so a session's transcript is a bf16 re-draw of the same class as the batch preset's
row-count draws. Measured on `nemotron-mt-td-x1d` (qjjoelpq/32z5012): eval-30 streamed sequentially (B = 1) =
**29.27**; the same 30 files streamed while 40 / 160 other real-time sessions were live = **29.45 / 29.30**
(+0.18 / +0.03 macro), 0/30 finals byte-identical to the sequential run, per-file |Δ| mean 1.3 / max 8.4 — the
band the `MT_XDIAR=1` vs `0` comparison also lands in (1.39 / 11.3). Macro-stable under load, per-file
re-drawn; the bit-exact combination is `MT_XDIAR=0` with a single live session.

Inductor is involved only through FlexAttention inside the Sortformer core (the ASR encoder graphs and the
RNNT decoder graphs replay eager kernels). The batch preset always runs the diarizer at 8 rows, so the
bs=1-specialised FlexAttention graph the standalone diarizer fork found (+0.65 DER at bs=1) never executes
there; `MT_XDIAR=0` runs it at 1 row for both reference and serving, `MT_XDIAR=1` at B rows. Measured on
`nemotron-mt-td-x1e` (eval-30, same replica, control flips): padding the diarizer to ≥ 2 rows is a *worse* draw
in both modes — mode 0 1-row 29.48 → 2-row (`MT_DIAR_PAD_ROWS=2`, row duplicated) **30.61**; mode 1 B free
29.27 → B ≥ 2 (`MT_XDIAR_ROWS=2`) **30.41** — with more insertions and deletions and a wider per-file spread
(mean |Δ| 2.1–2.8, max 16–21, vs 1.39 / 11.3 between the two 1-row modes). The bs=1 kernel is not a bias against
this pipeline; both pad knobs stay off and `MT_XDIAR=1` with B free is the default.

### Where the tick's time goes (live `{"stats":1}`, N=160, `MT_XDIAR=1`)
Stepper busy **0.55–0.79** over the arm (the audited build: 0.94 at the same load), tick p50 17.6 / p99
29 ms, B p50 2–3 (the stepper keeps up, so batches stay small). Per tick: ASR 8.5 ms (encoder graph at the
8/16-row bucket + label-looping decoder + per-row hypothesis bookkeeping), fused diarizer 4.8–5.1 (was
3.8 × B), mel 3.5, gather 0.4, seglst 0.3, cache write-back 0.2.

### 20-min arms (k6 in-cluster, `nemotron-mt-td-x1c` 3yv622e3/w55opjp, generator 3.3 / 4.3 cores mean at 160 / 190)

| streams | done | emit lag p50 / p95 / p99 | window p95 (w0 → w1) | backlog p95 (w0 → w1) | final latency p50 | verdict |
|---|---|---|---|---|---|---|
| **160** | 143/160 (17 gateway resets, 0 server errors) | 0.31 / 0.43 / 0.49 s | 0.39 → 0.47 s | 0.38 → 0.40 s | 0.21 s | **holds** (the audited build at 160: 0.25 → 0.70 s rising every window, stepper 0.94) |
| **190** | 189/190 (1 gateway reset, 0 server errors) | 0.32 / 0.46 / 0.76 s | 0.41 → 0.49 s | 0.40 → 0.44 s | 0.50 s | **holds** (the audited build collapsed at 190 in a cold 300 s run) |

At 190 the stepper is not the wall yet: live stats mid-arm showed busy 0.79 with B still 1–5 and tick p50 21.6 ms,
i.e. arrivals are spread thinly enough that batches barely form; the remaining budget is shared between the
stepper's Python (~40–50 % of a core) and the connection side of the GIL (~1,900 inbound 100 ms frames/s of JSON +
base64 + numpy, ~170 outbound replace-style partials/s of 40–90 KB ≈ 10 MB/s through the websocket stack). The
end-of-arm commit burst (all sessions finalising within one second) is the only >1.5 s event: 575 frames across
168 streams, max 2.5 s, at t = 1202 s; final latency p95 2.1 s. The analyzer's own end-backlog clause flags that
burst as "degraded"; under the degradation rule used here (no 10-min window p95 > 1.5 s, no backlog growth) both
arms hold.

### Hour-long 190 (k6 in-cluster, 60 min of audio per stream, `nemotron-mt-td-x1d` qjjoelpq/32z5012, generator 5.0 cores mean / 6.6 max)

| | 190 streams (loaded replica) | 1-VU idle control (same build, `nemotron-mt-td-x1e`, last 45 min) |
|---|---|---|
| emit lag p50 / p95 / p99 (hour) | 0.34 / 0.51 / 0.61 s | — |
| window median, 10-min windows 0 → 5 | 0.24 / 0.33 / 0.34 / 0.34 / 0.35 / 0.35 s | 0.23 / 0.32 / 0.32 / 0.32 / 0.35 s |
| window p95, 10-min windows 0 → 5 | 0.39 / 0.48 / 0.49 / 0.51 / 0.52 / 0.56 s | 0.39 / 0.46 / 0.47 / 0.49 / 0.50 s |
| completed | 149/190 — 0 server errors (no `ws session failed`, no error frames), 41 gateway socket resets (`WebSocketDisconnect`, clustered at min 53 and 58) | 1/1 |
| finals | p50 3.2 s (synchronized commit burst of 149 sessions) | — |
| server at the end of the hour | stepper busy 0.88 cumulative (incl. the earlier gate runs), tick p50 20.7 / p99 88 ms, GPU 4.4 GB allocated / 10 GB reserved, flat | — |

The loaded replica is within 0.01–0.06 s of the idle control in every window: the staircase is the replace-style
partial's wire size, the server adds tens of milliseconds at 190. **Verdict: 190 hour-long real-time streams per
RTX-PRO-6000 hold (every window p95 ≤ 0.56 s, no backlog growth, 0 server errors).** The audited build collapsed at
190 in a cold 300 s run.

### 20-min screens above 190 (`nemotron-mt-td-x1f` qrp95or3/qkj9lol, 1-VU idle control on `nemotron-mt-td-x1f-ctl`, same build)

| streams | done | emit lag p50 / p95 / p99 | window p95 (w0 → w1) vs control | backlog p95 (w0 → w1) | final p50 | verdict |
|---|---|---|---|---|---|---|
| **220** | 211/220 (9 gateway resets, 0 rejected, 0 server errors) | 0.32 / 0.47 / 0.72 s | 0.40 → 0.50 s vs 0.38 → 0.46 s | 0.40 → 0.48 s | 0.29 s | **holds** (analyzer verdict: not degraded) |
| 250 | 229/250 (21 gateway resets, 0 rejected, 0 server errors) | 0.32 / 0.48 / **1.10** s (max 33.7) | 0.41 → **0.57** s vs 0.39 → 0.44 s (control reset at min 15) | 0.40 → 0.56 s | 1.08 s | holds under the rule, **at the edge**: the tail opens (p99 1.1 s), the server's share grows to 0.13 s in w1; the analyzer flags the commit burst (end-backlog excess 1.6 s, as at the hour-long 190) |

`predict_concurrency: 220` (20-min screen with control; 190 is the hour-long confirmation), `concurrency_target`
~190. The generator ran 5.3 cores mean / 6.9 max of 16 at 220; the frame log is compact (`mt_stream_k6.js` logs a
lag record per partial and a full frame only every 100th). The lever after this is the stepper in its own process
(design in `XSESSION.md`): at 190 the stepper is 0.75–0.88 busy with B 1–5 and shares the GIL with ~1,900 inbound
frames/s and ~10 MB/s of partials.

## Cold start / first connection (RTX-PRO-6000, weights mounted)

`model.load()` ~50 s: checkpoint restore, ASR warm at k=8..1 with injected speaker targets (encoder graph,
decoder state sized once — white-noise warm sessions never activate a speaker, so the eager build's first
connection paid ~1 s of RNNT graph capture and 2 s of flex-attention recompiles), then a 180 s synthetic
diarizer session (FIFO fill → pop → compression → every steady-state length, ~150 graph captures).
Idle GPU after load: ~2.6 GB allocated / ~6 GB reserved; ~+0.1 GB per live stream. First-connection
numbers for the registry copy are in `README.md`.

## Scorer note
Streaming and batch eval-30 figures (29.27 / 29.48 here, 31.90 in `../batch/BENCHMARK.md`) come from the same
scorer: `stt-benchmark/.venv` (`whisper_normalizer` importable), `stt_benchmark.diarization.scoring.cp_wer` on
`normalize_seglst`, references from `data/notsofar_eval/seglst`. Cross-check 2026-09-17: the batch preset's stored
hypotheses scored through the streaming harness's `score()` give 31.899, the batch harness's own number; the
stored streaming finals re-scored give 29.48 / 29.27. The 2.4-point gap between the presets is numerics — the
streaming ASR runs bf16 at 8/16/32-row buckets over a per-step mel with a per-session diarizer, the batch ASR
at a pinned 32 rows over a whole-file buffer with 8-session diarizer slabs — i.e. different draws from the
documented per-file band (SD ≈ 5.7 per session), measured identically by the audit fork a day earlier.

All cpWER here is from the canonical scorer: `stt-benchmark/.venv` (Whisper `EnglishTextNormalizer` via
`whisper_normalizer`, meeteval 0.4.3, `stt_benchmark.diarization.scoring.cp_wer`). An environment without
`whisper_normalizer` silently falls back to a naive normaliser and shifts every figure ~2 pt — do not
compare across scorers.

## Reproduce
```
V=stt-benchmark/.venv/bin/python; export MT_MID=<mid> MT_DID=<did>
# correctness (3 whole files; run twice and compare "finals" for determinism)
$V rsi-bench/nemotron_diar/mt_fast_client.py cpwer --secs 0 --out out/run
# concurrency curve (real time, 90 s clip)
$V rsi-bench/nemotron_diar/mt_fast_client.py load --secs 90 --n-list 15,20,30,40,60,80 --out out/curve
# first-connection timing and malformed-frame probes
$V rsi-bench/nemotron_diar/mt_registry_probe.py first; $V rsi-bench/nemotron_diar/mt_registry_probe.py bad
```
Per-phase budgets need `MT_PROFILE=1` on the replica (synchronize-bounded timers attached to every message).
Raw data: `rsi-bench/nemotron_diar/results_mt_stream/` (`FASTSTEP.md`, `fast*/`, `RESULTS.md` for the eager build).

## Robustness audit (2026-09-16 overnight, registry HEAD on fde-internal, RTX-PRO-6000)

Re-run any of these with `rsi-bench/nemotron_diar/smoke_td.py --target MID/DID` (canonical scorer venv).

- **Memory under churn — no leak.** GPU memory plateaus at **~12.7 GB allocated / ~19.6 GB reserved**
  after the diarizer's per-length graphs and the RNNT decoder's per-bucket graphs are all captured
  (~6 full sessions), then stays flat: measured dead-flat over sessions 8→14 of a full-file churn and
  over **136 sessions total including 50 abrupt mid-session TCP resets** (Δallocated **+0.00 GB**,
  `live` returns to 0 each time). The transient OOM seen mid-audit was the **`MT_PROFILE=1` measurement
  build only** (the profiler path); the shipped config (no profiler) does not leak. `MT_GC_EVERY` runs a
  defensive `gc.collect()+empty_cache()` every 4 ended sessions. A `{"stats":1}` handshake returns the
  live count, `gpu_alloc_gb`/`gpu_reserved_gb` and stepper/detok counters for monitoring.
- **`words` inner-space glue fixed.** NeMo emits a bare word-marker (`▁`) as its own piece before some
  words; the piece-join rule glued it to the previous group, so a `words` entry read `"good ideas"` with
  an inner space. Now words split at a lone `▁` (and a trailing-marker flag carries across steps), and any
  group that still decodes to whitespace is split. **Per-speaker word sequences and segment text are
  unchanged**; 3-file FINAL cpWER **23.57 / 28.38 / 18.45 unchanged**, 0 words-with-space and 0
  timing/monotonicity/text-consistency violations across the 3 files (`smoke_td.py words`).
- **Overlap flags accurate.** On MTG_32315 (RTTM overlap-heavy) the `overlap`-flagged segments have
  **precision 0.919 / recall 0.989** against reference overlap regions ≥ 0.5 s (`smoke_td.py overlap`).
- **Error paths.** 8 malformed-frame classes (non-JSON, non-object, bad base64, odd PCM, `max_speakers`
  99 / "abc" / bool, non-string audio) each return an `{"type":"error"}` frame + clean 1000 close; an
  unknown frame type is ignored and the session still finalizes; commit-with-no-audio, commit-after-5 s-
  silence, commit-as-first-frame, and speech+silence+commit all finalize correctly; two connections with
  the same `session_id` run independently (`smoke_td.py malformed/commit/dup/k`).
- **`max_speakers` handshake.** k=1/2/8 all finalize with hypothesised speakers within the cap.

## `MT_XDIAR=1` decision — 30-file eval

Both modes on the **same deployment**, canonical scorer, NOTSOFAR eval-30:

| mode | macro cpWER | vs XDIAR=0 | per-file stability |
|---|---|---|---|
| `MT_XDIAR=0` (default, per-row sync diarizer) | **29.48** | — | reference |
| `MT_XDIAR=1` (batched async-layout diarizer) | **29.27** | **−0.21 (within the 0.5 gate)** | mean \|Δ\| 1.4, **max \|Δ\| 11.3** (MTG_32009 −11.3, MTG_32026 +7.9) |

Macro is a wash (−0.21, better), so XDIAR=1 clears the quality gate; but it is **not output-preserving**
per file (async bf16 numerics re-draw the speaker gate — the same shape noise as `results_mt_td/
B_GT_1_ROOTCAUSE.md`, SD ~5.7 per session). Its value is throughput: the diarizer term drops from
3.8 ms × B per tick to ~5 + 0.9 × B, which is the lever to push the hour-long stream ceiling past the
lock/sync-diarizer 160. Ship it as default only paired with the capacity ladder that shows the win
(`rsi-bench/nemotron_diar/results_mt_stream/audit/`, `D_x0`/`D_x1`).
