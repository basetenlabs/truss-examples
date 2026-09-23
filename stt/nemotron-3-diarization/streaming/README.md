# Nemotron 3 Diarization (streaming)

> Streaming WebSocket preset for NVIDIA's [`nvidia/Nemotron-3-Diarization`](https://huggingface.co/nvidia/Nemotron-3-Diarization)
> (General Access checkpoint), released under the [OpenMDW 1.1 license](https://openmdw.ai/license/1-1/)
> (commercial use permitted). The `.nemo` is pulled at deploy from that repo; it is not committed here.
> Sibling presets: `../batch` (recorded audio over HTTP) and
> `../../nemotron-3-diarized-transcription/streaming` (speaker-tagged words).

## What it is
Live speaker diarization over a WebSocket. The client streams PCM16 mono 16 kHz audio
chunk-by-chunk; the server drives NeMo's streaming Sortformer loop with **per-connection state**
(the AOSC speaker cache + FIFO queue) and emits speaker turns incrementally. ≤ 8 speakers, labels
ordered by arrival. Speaker identity is carried forward across chunks — no re-clustering, no
retroactive relabeling.

## Request-level latency
Three algorithmic-latency profiles are **pre-loaded as separate model instances**, and each
connection picks one in its handshake (`NEMO_DIAR_PROFILES`; `verylow` is measured but not offered by
default — see below). The streaming knobs (`chunk_len`, `chunk_right_context`, `fifo_len`,
`spkcache_*`) live on shared module state, so one instance cannot serve mixed profiles concurrently;
per-connection state is passed into each step, so many connections share a profile's instance safely.

| Profile | Input-buffer latency | Notes |
|---|---|---|
| `low` | 1.04 s | recommended real-time |
| `ultralow` | 0.32 s | lowest latency; 3× the step rate of `low` |
| `offline` | 30.4 s | best DER; high-latency streaming |
| `verylow` | 0.64 s | *measured, not offered by default*; opt in with `NEMO_DIAR_PROFILES=offline,low,verylow,ultralow` |

Buffer latency = `(chunk_len + chunk_right_context) × 80 ms`. Compute runs far faster than
real time, so the buffer dominates the time-to-label.

## Wire protocol
Client → server (text frames, JSON):
```json
{"latency": "low"}                                              // optional handshake (first frame)
{"type": "input_audio_buffer.append", "audio": "<base64 pcm16 @16kHz mono>"}   // repeat
{"type": "input_audio_buffer.commit"}                          // finalize + flush + close
```
Server → client:
```json
{"type": "diarization", "is_final": false, "processed_s": 12.3, "num_speakers": 2,
 "turns": [{"start": 0.51, "end": 12.62, "speaker": "speaker_0"}, ...]}   // replace-style partials
{"type": "diarization", "is_final": true, "is_end_of_audio_flush": true, "turns": [...]}
```
Partials are replace-style: each emission carries the full current turn list (client replaces its
view). `threshold` (per-speaker activity, default 0.5) is settable in the handshake. Malformed frames
(bad JSON, unknown `latency`, invalid base64, `threshold` outside (0, 1)) get an `{"type": "error"}`
frame and a clean close; server faults are logged with a traceback and close the socket with an error
frame.

## Truss shape / how it serves many streams
Same container recipe as the batch preset (`nvcr.io/nvidia/nemo:26.08` + ASR deps + NeMo overlay pinned
at `3c2d62ae7eb4`; see `../batch/README.md`), served as a WebSocket endpoint. RTX-PRO-6000.

- **Cross-session batched stepper, fused tick** (`packages/xsession.py`, `NEMO_DIAR_XSESSION=1`,
  `NEMO_DIAR_XSESSION_TICK=fused`): one thread per profile takes every connection's ready chunk and runs
  ONE CUDA-graph replay per tick for all of them, covering the whole step — mel of the raw windows, gather
  of the rows' streaming state from preallocated **slot tables** (a row per live session), pre-encode →
  concat+pad → FastConformer → Sortformer head, NeMo's async cache/FIFO update with the speaker-cache
  compression computed for every row and selected by `where` (NeMo's own path decides it with a host sync),
  and the write-back by slot index. Inputs and outputs travel through pinned buffers, and ticks are
  pipelined one deep: the thread packs and issues tick k+1 while the GPU runs tick k, then waits on an
  event with the GIL released. Per tick the thread holds the GIL for ~1.3 ms (was ~9 ms), so at 250
  streams the stepper costs ~7 % of a core (was 75–94 %). Ticks coalesce to min(16, live sessions) rows or 40 ms, where the
  per-row GPU cost bottoms out (a lone stream never waits). The encoder is `torch.compile`'d (Inductor kernels captured in the graph)
  and the core runs in fp32 (bf16 is available behind `NEMO_DIAR_CORE_DTYPE=bf16`; it doubles the
  per-file DER spread vs the batch path for the same set-level DER, so it is not the default). Connections never touch the GPU: the handler slices the raw window a
  chunk needs (`normalize: NA` mel is frame-local, so a trailing window with 3 frames of left context is
  exact) and awaits the tick.
- **Per-connection sender mailbox**: outbound frames go through a one-slot mailbox drained by a sender
  task, so a slow reader of the growing replace-style partials (hour-long calls reach ~135 KB per
  partial) never stalls that connection's inbound audio; an unsent partial is superseded by the next,
  finals and errors are always delivered (`partials_coalesced` in the session log).
- **Incremental turns with a committed prefix**: per-session run-length state on the CPU updated from each
  chunk's predictions; every closed run whose start precedes all still-growing runs is appended once to a
  committed JSON prefix, so a partial costs O(uncommitted turns) however long the call (same output as
  re-thresholding the whole history, byte-identical on 26 k random partials).
- A/B paths behind flags: `NEMO_DIAR_XSESSION_TICK=legacy` (the previous stepper tick: per-session tensors,
  core-only graph, NeMo's eager update), `NEMO_DIAR_FUSED_COMPILE=0` (eager kernels), `NEMO_DIAR_CORE_DTYPE`
  (fp32 | bf16); `NEMO_DIAR_XSESSION=0` restores the per-connection paths (sync-mode state with the eager
  core captured per sequence length, `NEMO_DIAR_SYNC_GRAPHS=1`, bit-identical to NeMo's sync path; or the
  original `torch.compile`'d encoder, `NEMO_DIAR_COMPILE=1`).

## Concurrency, autoscaling, cold start (measured — see `BENCHMARK.md`)
- **Capacity:** one RTX-PRO-6000 holds **560 hour-long real-time `low` streams** (60-min ladder, in-cluster
  k6, every 10-min window's p95 emit lag ≤ 0.70 s with the p50 equal to the idle control's, zero backlog
  growth; 320 / 400 / 480 the same; 640 collapses GPU-bound at ~890 rows/s) and **200 `ultralow` streams**
  (p95 ≤ 0.30 s, GPU 76–81 %; 250 collapses). Re-confirmed on the GA checkpoint with 20-min screens (560
  `low`: worst-window p95 0.58 s vs control 0.53 s, 0 rejected; 200 `ultralow`: p95 0.26 s); the GA `low`
  profile's chunk 9 (+rc 4) means fewer, larger ticks, so emit-lag p50 at 560 is 0.46 s (preview 0.30 s)
  while the 1.04 s buffer latency is unchanged. The stepper thread holds the GIL for ~0.1 core at 560 (stepper
  v1: 94 % busy at 250); the ceiling is the GPU's 1.0 ms per row. That is 44 % / 46 % of NVIDIA's lockstep
  bf16-compiled reference throughput on the same GPU, with an fp32 core under the real-time rule.
- **`predict_concurrency: 560`** (config): the measured hour-long hold; excess connections are rejected
  rather than dragging every stream into collapse (truss admits WebSocket sessions through this semaphore:
  a blocked session gets no frames at all, so size it to the measured hold, not above).
- **Autoscaling for a library listing:** min 1 replica (cold start is too long for scale-from-zero on a
  live-audio product), **`concurrency_target` 400** WebSocket sessions per replica (400 and 480 hold at the
  control's latency; the band to 560 absorbs bursts and gateway reconnects), so 800 streams ⇒ 2 replicas.
  `ultralow` traffic is 3× the density — a mostly-`ultralow` deployment should cap at ~200.
- **Per-profile admission (the mix rule):** `predict_concurrency` is a global WebSocket cap and cannot see
  which profile a session asks for, but an all-`ultralow` replica saturates its GPU at ~250 sessions while
  560 `low` sessions hold. The server therefore weighs live sessions in `low`-row-equivalents against one
  GPU budget — `low` 1, `ultralow` `NEMO_DIAR_MAX_LIVE_LOW / NEMO_DIAR_MAX_LIVE_ULTRALOW` = 560/200 = 2.8,
  `offline` 0.2 — and a handshake that would exceed the budget (560) gets `{"type":"error","error":"capacity:
  …"}` and a clean close instead of degrading every stream; clients retry on another replica. Any mix that
  satisfies `low + 2.8·ultralow + 0.2·offline ≤ 560` is admitted (e.g. 280 `low` + 100 `ultralow`).
- **Cold start:** `load()` 130–160 s (restore 3 instances, warm session + 16 whole-step graphs per profile,
  the `offline` encoder's Inductor compile ~100 s dominates; b10cache keeps the compile cache for restarts).
  `startup_threshold_seconds` 600.
- **Knobs:** `NEMO_DIAR_XSESSION_B` (graphs per batch size, 32; `_B_OFFLINE` 8), `NEMO_DIAR_XSESSION_B_TARGET`
  (16), `NEMO_DIAR_XSESSION_WAIT_MS` (40), `NEMO_DIAR_XSESSION_SLOTS` (640), `NEMO_DIAR_PROF` (per-step records
  + GPU samples in every frame), `NEMO_DIAR_BENCH` (load-time tick bench + fused-vs-legacy self-check).

## Status / correctness
- Fused tick vs the previous stepper's tick with NeMo's own update, same model, synthetic session through
  cache fill / FIFO pops / compression: **max |Δ| = 0, identical turns** on all three profiles (load-time
  self-check, `NEMO_DIAR_BENCH=1`).
- Streaming output vs the whole-file batch endpoint, 10 NOTSOFAR files, DER@0. Preview build / preview
  weights (stepper v1, fp32 eager): `low` mean 0.36, `offline` 0.19, `ultralow` 1.00; compiled encoder:
  `low` 1.17 / max 2.22, `ultralow` 0.84 / max 2.54. **GA weights** (this build, fp32 core vs the batch
  preset's bf16 core): `low` 2.13 / max 6.48, `ultralow` 2.78 / 4.95, `offline` 1.26 / 4.43 — both sides
  score the same against the reference; the wider band is the dtype mismatch between the two presets and is
  to be re-measured with matched dtypes before a tighter acceptance band is restated.
- **Full-set NOTSOFAR-129 via the WebSocket at `low`, 6 concurrent streams:** stepper v1 18.09 / 12.31;
  this build fp32 eager 18.27 / 12.50, fp32 + compiled 18.18 / 12.41 (shipped config 18.07 / 12.31 on
  the 126 files that returned), bf16 18.15 / 12.37 — all inside the 0.3
  acceptance band (the spread is which rows share a tick, not the kernels).
- **Not done:** reconnect/resume; a delta (non-replace) partial protocol — the replace-style partial is
  the largest per-connection cost on hour-long calls and the likely trigger of the gateway's connection
  resets; neighbour-independent numerics under batching (rows sharing a tick round differently from a
  solo run — intrinsic to batching this model, see `BENCHMARK.md` §2; exact per-row output only
  unbatched via `NEMO_DIAR_SYNC_GRAPHS=1`).

## Release state (General Access checkpoint, 2026-09)
The GA checkpoint (`nvidia/Nemotron-3-Diarization`, OpenMDW-1.1) replaced the early-access preview on
2026-09-21: retrained bf16 weights (189 MB, sha256 `867c53f5…`), speaker cache 264 frames, the same
latency profiles this preset already shipped (`low` chunk 9 + rc 4). Loads on the pinned NeMo overlay,
16 whole-step graphs per profile capture as before, the 560 `low` / 200 `ultralow` holds and the
admission budget stand (`rsi-bench/nemotron_diar/release/GA_VALIDATION.md` §5). Not yet re-run on GA:
the hour-long ladder arms and the `NEMO_DIAR_BENCH=1` fused-vs-legacy self-check.