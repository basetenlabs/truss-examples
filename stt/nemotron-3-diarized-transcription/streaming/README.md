# Nemotron 3 Diarized Transcription (streaming)

> **English-only.** Streaming WebSocket preset for NVIDIA's coupled overlap-aware speaker-attributed
> transcription: the diarizer [`nvidia/Nemotron-3-Diarization`](https://huggingface.co/nvidia/Nemotron-3-Diarization)
> (General Access checkpoint, [OpenMDW 1.1](https://openmdw.ai/license/1-1/), commercial use permitted)
> drives one ASR instance per speaker of `nvidia/multitalker-parakeet-streaming-0.6b-v1` (NVIDIA Open
> Model License). Both checkpoints are pulled at deploy from Hugging Face, never committed.
> Sibling: `../batch` (recorded audio over HTTP); diarization only: `../../nemotron-3-diarization`.

## What it is
Live "who said what", overlap-aware, over a WebSocket. The client streams PCM16 mono 16 kHz
audio; the server runs NVIDIA's `SpeakerTaggedASR` integration per connection — Nemotron-3
(streaming Sortformer) emits per-speaker activity and **multitalker-parakeet-streaming-0.6b-v1**
runs **one ASR instance per speaker** on the *same mixed audio* (speaker-kernel injection, no
enrollment) — and emits speaker-tagged segments after every ~1.12 s ASR chunk. Same models,
knobs and chunk geometry as `../batch` (att_context `[70, 13]`, diarizer
`spkcache 264 / fifo 264 / update 222`, `parallel_speaker_strategy`, `cache_gating`,
`binary_diar_preds`, bf16); the difference is that audio arrives incrementally and each
connection keeps its own ASR caches + diarizer state. Mel is computed once per chunk on the raw
tail with a few frames of left context over whole STFT frames only, so features match the
whole-file path; the step itself is dispatched through CUDA graphs (see "Performance build").

## Wire protocol
Client → server (text frames, JSON):
```json
{"session_id": "call-42", "max_speakers": 8}                                    // optional handshake (first frame)
{"type": "input_audio_buffer.append", "audio": "<base64 pcm16 @16kHz mono>"}   // repeat (100 ms chunks work well)
{"type": "input_audio_buffer.commit"}                                          // finalize + flush + close
```
Server → client:
```json
{"type": "transcription", "is_final": false, "session_id": "call-42", "processed_s": 12.32,
 "num_speakers": 2,
 "segments": [{"speaker": "speaker_0", "start": 0.51, "end": 4.2, "text": "so the agenda today"},
              {"speaker": "speaker_1", "start": 3.9, "end": 12.1, "text": "right, first item"}]}
{"type": "transcription", "is_final": true, "session_id": "call-42", "processed_s": 363.4, ...}
{"type": "error", "error": "max_speakers must be in [1, 8]"}                  // then the socket closes
```
Frames are replace-style: each emission carries the full current segment list (the client
replaces its view). `max_speakers` (1–8, default 8) caps the per-speaker ASR instances for the
connection. Speaker labels are session-local and arrival-ordered — not identities.

### Turns and partials (`MT_TURN_SEGMENTS`, `MT_PARTIALS`, default on)
NeMo's own SegLST keeps ONE running sentence per speaker and appends every step's new words to it; a
new sentence starts only after that speaker's own 30 s silence (`sent_break_sec`), so other speakers
never break a block — MTG_32000 (6 min, 4 speakers) came out as 12 segments of ~100 words, 8 of them
spanning another speaker's turn. `packages/mt_turns.py` re-cuts the same words into turns from the
RNNT emission frames (word start/end at 0.08 s). Each speaker has its own open tail; it **closes after
that speaker's pause > `MT_TURN_PAUSE_S` (1.2 s)**, at its **sentence-final punctuation**, and on a
**hand-over** — since its last word other speakers have talked for ≥ `MT_TURN_HANDOVER_S` (1.0 s) or
≥ `MT_TURN_HANDOVER_WORDS` (3) words. A one- or two-word backchannel is below both, so it never splits
the running turn; it becomes its own short segment overlapping it. Segments may therefore overlap in
time (`overlap: true`, `overlaps_with`), and `segments` is ordered by start so parallel turns sit
next to each other.
```json
{"type": "transcription", "is_final": false, "processed_s": 12.32, "num_speakers": 2,
 "segments": [{"speaker": "speaker_0", "start": 0.51, "end": 4.2, "text": "So the agenda today.",
               "overlap": true, "overlaps_with": ["speaker_1"],
               "words": [{"w": "So", "start": 0.51, "end": 0.59}, ...]},
              {"speaker": "speaker_1", "start": 2.9, "end": 3.3, "text": "Mm-hmm.",
               "overlap": true, "overlaps_with": ["speaker_0"], "words": [...]}],
 "partial":  [{"speaker": "speaker_0", "text": "and then the", "start": 4.4, "end": 5.1},
              {"speaker": "speaker_1", "text": "right, first item", "start": 4.6, "end": 5.9}]}
```
- `segments` = closed turns (immutable apart from a rare cross-chunk word continuation and an `overlap`
  flag that can flip to true when a later-closing parallel segment touches it); `partial` = every
  still-open tail (one entry per speaker currently talking, ordered by start), re-sent every chunk
  (~1.12 s) until it closes and moves into `segments`. The final frame has every word in `segments`
  and `partial: []`.
- `MT_TURN_OVERLAP=0` (or `"overlap": 0` in the handshake) restores the single-tail cut: one open
  segment in total, closed by any other speaker's word — simultaneous speech then comes out as
  one/two-word alternation and `partial` has at most one entry; `overlap`/`overlaps_with` are absent.
- `words` per segment costs ~3x frame bytes; a session can drop it with `"words": 0` in the handshake
  (`turn_segments` / `partials` / `overlap` are per-session overrides too).
- Identity: per speaker the words are exactly NeMo's (built from the same token ids via the
  `IncrementalDetok` piece cache, no re-decode); 3-file FINAL cpWER unchanged at 23.57 / 28.38 / 18.45.
  `MT_TURNS_VERIFY=1` logs `TURNS verify ... identical=N` at every final.
- Step cost: the step path records three ints per speaker per step (~1 µs); words and turns are built in
  `message()` off the step path.

## Per-step contract
- One server step per full ASR chunk (112 mel frames ≈ 1.12 s of audio); a partial is emitted
  only when a step ran, so a client sending 100 ms chunks sees one partial per ~11 appends.
- `processed_s` is the audio time the emitted segments cover. Emit lag relative to live audio
  is the chunk buffer (~1.1 s) plus queueing behind other sessions' steps (see status).
- `commit` zero-pads the tail to a whole chunk, runs the last step with the buffer empty (all
  pending hypotheses committed), sends the final and closes.
- Malformed frames (non-JSON, non-object, invalid base64, odd PCM byte count, `max_speakers`
  outside 1–8) get an `{"type": "error"}` frame and a clean close; server faults are logged
  with a traceback and, best-effort, an `internal error` frame.

## Performance build
The eager NeMo step was CPU launch-bound: ~3,100 kernel launches for ~11 ms of GPU work, 53 ms of
lock-held time per 1.12 s chunk, 15 streams per GPU. `packages/mt_fast.py` dispatches the same NeMo
calls with a handful of graph replays instead, **~13 ms per step** (60 streams per replica on the lock path;
the cross-session stepper below carries 190–220). Every lever
is an `MT_*` env knob (defaults = the validated combination, see `config.yaml`):

| knob | what it does | effect |
|---|---|---|
| `MT_MEL_PER_STEP` | log-mel once per 112-frame step from a bounded ring (same frames, same per-chunk normalisation) | mel 12 → 1 ms; identical transcripts; removes the N>1 GIL/stream interference |
| `MT_ASR_DTYPE` | ASR encoder weight dtype: `bf16` (default), `fp32` (fidelity option), `autocast` (the eager path) | bf16: no per-step weight casts, encoder graph 5 ms; fp32: −0.09 vs −0.34 cpWER at 20 ms/step |
| `MT_ENC_CUDA_GRAPHS` | NeMo's `CudaGraphsStreamingEncoderStep` (pre-encoder inside the graph, static speaker-target buffer) | encoder 21.5 → 5 ms |
| `MT_DIAR_GRAPHS` | sync-mode diarizer core captured in a CUDA graph per sequence length (~150 graphs, captured at load) | diarizer 21.7 → 3.8 ms, bit-identical to NeMo's sync path |
| `MT_PAD_ROWS` | constant 8 speaker rows per ASR step | deterministic transcripts, one encoder graph, fixed decoder state; +1 ms |
| `MT_NO_DEEPCOPY` | shallow-copy hypotheses (the decoder returns a cloned state anyway) | gather 0.9 → 0.26 ms |
| `MT_INC_DETOK` | detokenise only the new tail of each speaker's hypothesis (committed prefix cached on the hypothesis, cut at word starts) instead of NeMo's whole-session re-decode per step | step flat over an hour (13.6 → 13.6 ms; was 13.9 → 15.9–17.2); text byte-identical, `MT_INC_DETOK_VERIFY=1` checks every step |
| `MT_WARMUP_SECS` | length of the load-time diarizer warm session (180 s walks every sequence length) | no graph captures under traffic |
| `MT_DIAR_ASYNC`, `MT_DIAR_COMPILE`, `MT_TF32` | rejected levers kept for A/B (off) | not output-preserving (+1.1 / +5.0 / −0.82 cpWER) |

Warm-up drives the ASR path at k=8..1 with injected speaker targets (white noise never activates a
speaker, so the eager build's warm sessions never touched the ASR) and then a 180 s diarizer session.

## Numerical determinism
bf16 cuBLAS GEMMs on RTX-PRO-6000 pick split-K kernels whose rounding depends on the output tile and on
the batch M-dimension (one bf16 ulp, first visible in the encoder's `linear_q`); the 24-layer conformer and
greedy RNNT amplify that into different words (`rsi-bench/nemotron_diar/results_mt_td/B_GT_1_ROOTCAUSE.md`).
In the eager build every change of the active-speaker count re-rolled the shapes for every speaker's
stream. This build pads every ASR step to 8 rows, so one connection produces one reproducible transcript
(two passes byte-identical). The −0.34 macro cpWER vs the eager build (23.47 vs 23.81 on 3 files) is a
different draw from that same rounding noise (per-session SD 5.7, macro band ±0.8), not a quality change;
`MT_ASR_DTYPE=fp32` is the closest-to-eager option (−0.09) at 20 ms/step. The diarizer is untouched
numerically (graph replay of the eager sync kernels); async / compiled diarizer variants were rejected for
exactly this reason (see `BENCHMARK.md`).

## Status
- **Correctness:** FINAL cpWER on 3 NOTSOFAR-1 eval sessions (MTG_32000/32003/32004, canonical scorer),
  preview diarizer: **23.57 / 28.38 / 18.45, macro 23.47**, deterministic (two runs byte-identical); eager
  build / `../batch`: 23.99 / 28.31 / 19.12 (23.81). **GA diarizer** (no code change): 30.58 / 22.74 / 15.78,
  macro **23.03** — per file the diarizer relabels differently, the macro is flat; the eval-30 batch number
  (28.58, `../batch`) is the one to carry. See `BENCHMARK.md` and "Numerical determinism".
- **Turn defaults (2026-09-22):** `MT_TURN_SENTENCE_BREAK=0`, `MT_TURN_PAUSE_S` 2.5, hand-over 2.0 s / 6
  words, `MT_TURN_FREEZE_CLOSED=1`. On a 150 s 4-speaker meeting this cuts 71 segments / 4 words median
  (23/70 consecutive same-speaker rows) to 34 / 12.5 (4/33) with identical words, and a closed turn's text
  never changes. `MT_TURN_DIAR_EOT_S` stays 0: closing on the diarizer's off-run lowered end-of-turn p50 to
  0.64 s but fragmented turns further. All are per-session handshake overrides.
- **Concurrency (one RTX-PRO-6000, real time, 100 ms chunks, 90 s clip):**

  | streams | lag p50 | lag p95 | backlog over the clip | lock-held step |
  |---|---|---|---|---|
  | 15 | 0.16 s | 0.28 s | +0.03 s | 12.5 ms |
  | 20 | 0.20 s | 0.35 s | +0.04 s | 13.3 ms |
  | 40 | 0.34 s | 0.62 s | +0.10 s | 13.5 ms |
  | **60** | **0.54 s** | **0.91 s** | **+0.09 s** — ceiling | 13.5 ms |
  | 80 | 1.27 s | 2.61 s | +1.36 s — falling behind | 13.5 ms |

  (History: the lock path's ceiling, `predict_concurrency` 64 at the time. The shipped stepper path holds 190
  hour-long / 220 in a 20-min screen — see "2026-09-17 build" below.) The step does not grow with N; GPU util
  at 60 streams is ~50% (median 67%). Memory is ~6 GB idle, +0.1 GB per stream.
- **Thread-safety, and why the design is one model pair + one lock:** NeMo's streaming step is not
  reentrant (the RNNT decoder's CUDA graphs and batched scratch state are shared). The first deploy
  (shared pair, no lock) corrupted concurrent sessions; 8 pairs with per-pair locks forced the graph
  decoder off and held only 10 streams. One pair + one replica-wide step lock is both the fastest and
  the only variant with zero session faults — and with a 13 ms step the lock is 70% occupied at 60 streams.
  The cross-session stepper (`MT_XSESSION=1`, `XSESSION.md`) keeps that single pair but runs every ready
  session's step in one pass per tick, which is what lifts the ceiling to 190–220.
- **Cold start (RTX-PRO-6000):** `model.load()` ~50 s with weights mounted — checkpoint restore, ASR
  warm (k=8..1, encoder graph capture, decoder state), 180 s diarizer warm session (~150 graph captures).
  DEPLOYING→ACTIVE ~60–90 s; push→ACTIVE ~3–5 min with a cached image. `startup_threshold_seconds` is 300.
- **First connection after ACTIVE** (handshake + 3 s of speech sent at once + commit):
  connect 1.62 s, first `transcription` **1.83 s**, final 1.93 s (eager build: 1.07 / 2.87 / 2.98 s — the
  ~2 s first-connection premium was the ASR graph capture the white-noise warm-up never triggered).
  Second connection: 0.40 / 0.64 / 0.75 s.
- **Autoscaling for a library listing:** min 1 replica (a minute-plus cold start is not viable for
  scale-from-zero on live audio), `concurrency_target` ≈ 190 sessions per replica (latency flat to ~190 with
  the stepper 0.75–0.88 busy; the 220 cap absorbs bursts) so the next replica is warm before one fills;
  **190 hour-long streams ⇒ 1 replica** (64 on the lock path, 5 with the eager build).
- **Registry copy validated** on fde-internal (`wxpe087q` / `qkj46e8`, deactivated): loads from this
  layout in 35 s (1 encoder + 133 diarizer graphs captured, warm log `k_active=[8..1] exercised`),
  3-file FINAL cpWER 23.57 / 28.38 / 18.45, MTG_32000 byte-identical over two runs, N=60 real-time hold
  (60/60, lag p50 0.47 s / p95 0.85 s, backlog +0.15 s), all five malformed-frame cases → error frame
  and close 1000.
- **Long calls: flat.** NeMo re-detokenised the whole session per speaker per step (`Hypothesis.merge_`
  drops `text`, `decode_hypothesis` redoes tolist → `id_to_piece` → SentencePiece → punctuation regex), which
  grew the step 13.9 → 15.9–17.2 ms over a 60-minute session (all in the decoder phase, 2.96 → 5.02 ms).
  `packages/mt_incremental.py` decodes only the new tail from a prefix cached on the hypothesis: over the same
  hour the step is 13.64 → 13.57 ms (dec 2.77 → 2.71), every other phase flat. Output is byte-identical —
  3-file FINAL cpWER unchanged (23.57 / 28.38 / 18.45), identical seglst JSON on the 3 files and on the 60-min
  session (164 segments / 15,623 words), and 25,056 per-step decodes checked against NeMo's full decode with 0
  mismatches (`rsi-bench/nemotron_diar/results_mt_stream/LONGAUDIO.md`). Reconnect/resume is not implemented.
  Remaining levers: folding the diarizer's FIFO/cache update into its graph, cross-session batching of steps
  (shapes are now pinned, which it requires).
- **Hour-long streams per replica — history of the lock path** (k6 in-cluster, 60 min of audio per stream, real
  time): 20 / 40 / 60 held the whole hour (emit lag p50 0.22 s, every 10-min window p95 ≤ 0.36 s, backlog 0,
  step 12–13 ms, lock 22 / 42 / 69 % occupied); 80 collapsed (lock 97 %); 64 held (lock 74 %, window p95 ≤ 0.53 s)
  — the lock path's ceiling was ~64–70 streams. The cross-session stepper then carried 160 (94 % busy) and, on
  the 2026-09-17 build, **190 hour-long / 220 screened** (below). `BENCHMARK.md` "Long audio",
  `rsi-bench/nemotron_diar/results_mt_stream/sweep_2026-09-16/SWEEP.md`.

## Robustness & memory
GPU memory is stable under sustained call churn: it **plateaus at ~12.7 GB allocated / ~19.6 GB reserved**
once the diarizer's per-length and the RNNT decoder's per-bucket CUDA graphs are captured (~6 full
sessions), then stays flat — verified over 136 sessions including 50 abrupt mid-session disconnects
(Δallocated +0.00 GB; `live` returns to 0 each time). Do **not** enable `MT_PROFILE` in production (the
profiler path was the one config that grew memory under churn during the audit). A `{"stats":1}` handshake
returns `live`, `gpu_alloc_gb`/`gpu_reserved_gb` and stepper/detok counters. The full robustness suite —
malformed frames, commit paths, duplicate `session_id`, `max_speakers` handshake, word-timing/overlap
identity, disconnect leak-gate, hour-long single session — is `rsi-bench/nemotron_diar/smoke_td.py --target
MID/DID` (see `BENCHMARK.md` "Robustness audit"). `MT_XDIAR=1` is a throughput lever (batched diarizer),
macro-neutral on eval-30 (−0.21) but not output-preserving per file; see "`MT_XDIAR=1` decision".

## 2026-09-17 build (commits 096fc8b, 0ee34ee)
- Mel is computed in the stepper from the raw windows the connections hand it (one STFT per tick, the
  filterbank GEMM and normalisation per row so every row is bit-identical to the per-row mel); connections
  await an asyncio future (no parked pool thread per session, `MT_WORKERS` 32); partials are built from
  cached per-turn JSON fragments, byte-identical to `json.dumps`; one torch CPU thread.
- `MT_XDIAR=1` is the default: the diarizer runs as one CUDA-graph replay per tick over preallocated
  per-slot state tables (`MT_XSLOTS` 320). Eval-30 29.27 vs 29.48 for the bit-exact mode 0 (per-file
  |Δ| mean 1.4 / max 11.3 — NeMo's async numerics, the same draws as the audit); 3-file 23.64 / 28.02 /
  18.18. `MT_XDIAR=0` reproduces the committed build to the digit (23.57 / 28.38 / 18.45, eval-30 29.48).
- Capacity (k6 in-cluster, `BENCHMARK.md`): **190 hour-long real-time streams per RTX-PRO-6000 hold** — emit lag
  p50 0.34 / p95 0.51 / p99 0.61 s over the hour, every 10-min window p95 0.39–0.56 s within 0.01–0.06 s of the
  1-VU idle control, no backlog growth, 0 server errors (41 gateway socket resets counted separately); 20-min
  arms at 160 and 190 the same. 20-min screens with the idle control: **220 holds** (window p95 0.40 → 0.50 s vs control 0.38 → 0.46 s;
  211/220, 9 gateway resets); 250 holds under the rule but at the edge (window p95 0.57 s, p99 1.1 s, the
  server's share of the lag 0.13 s in the second window). The audited build collapsed at 190. `predict_concurrency` 220,
  `concurrency_target` ~190. Stepper 0.75–0.88 busy at 190 with B 1–5: the remaining budget is shared with the
  connection side of the GIL, which the stepper-in-a-process design in `XSESSION.md` would remove.
- Robustness: the RNNT label-looping decoder's `full_graph` mode (NeMo's default, conditional-node CUDA graphs)
  faults intermittently on its first replay after the encoder graph is captured: every load at 64 rows, some
  pods at 32 — a batch replica that had served 60 sessions failed every retry on reactivation, and one of the
  two hour-arm streaming replicas (`nemotron-mt-td-x1f`) crashed on its first load attempt in the 8/16/32-row
  warm and came up on the wrapper's retry. `MT_DEC_GRAPHS_MODE=no_while_loops` replays the same kernels under a
  host loop: identical bytes and step time on the batch preset (its default there). The streaming preset keeps
  NeMo's default because its tick budget and the 190-stream hour were measured with it; flip it before a
  long-lived deployment and re-check the tick p50 (expected within ~1 ms).

## Release state (General Access checkpoint, 2026-09)
The GA diarizer (`nvidia/Nemotron-3-Diarization`, OpenMDW-1.1) replaced the early-access preview on
2026-09-21 with no code or state-shape change (the speaker-kernel conditioning reads per-frame activity;
slab geometry 264/264/222 was already the GA geometry). Validated: load + 135 diarizer graphs, 3-file
streaming cpWER macro 23.03, eval-30 batch 28.58 (`rsi-bench/nemotron_diar/release/GA_VALIDATION.md` §6).
Autoscaling for the listing: min 1, `concurrency_target` ≈ 190, `predict_concurrency` 220. Not yet re-run
on GA: the hour-long 190-stream arm.