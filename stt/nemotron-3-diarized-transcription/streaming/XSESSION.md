# Cross-session batching (`MT_XSESSION`)

`packages/mt_xsession.py` replaces the replica-wide step lock with one **stepper thread**: every
connection whose 112-frame chunk is ready enqueues it, and the stepper runs all queued sessions in one
pass per tick — one gate, one ASR encoder graph replay + one batched RNNT decode over the (session,
speaker) rows of the tick (padded to the smallest of `MT_XSESSION_ROW_BUCKETS` = 8/16/32, one encoder graph each;
more rows run as 32-row slabs),
per-row cache write-back and per-session seglst/offset bookkeeping. Batches form only when arrivals
outpace the tick, so at low load B is 1 and the tick is the committed 13 ms step; under load the cost per
session-chunk falls to ~1.6 ms (B=26) instead of the queue growing. Full study, raw data and the design
notes: `rsi-bench/nemotron_diar/results_mt_stream/XSESSION.md`.

What NeMo's lockstep path could not do is done per row here: step-0 reset at admission, a per-session
chunk offset (NeMo shares one float), one `total_preds` tensor per session, `keep_all_outputs=False`
also on the final step (identical output for this geometry, lets it use the graph).

## Diarizer modes

| `MT_XDIAR` | what | numerics | cost per tick |
|---|---|---|---|
| **0 (default)** | the committed per-row sync-mode CUDA-graph path, one call per row | **bit-identical** to the committed build: 3-file FINAL cpWER 23.57 / 28.38 / 18.45 reproduced to the digit | 3.8 ms x B |
| 1 (experimental) | one batched call in NeMo's *async* streaming layout (fixed `[264\|264\|14]` rows with per-row lengths, core CUDA-graphed per B) | NeMo's async numerics: +1.6 cpWER macro on the 3 files (23.64 / 28.02 / 23.57), all from one speaker-gate flip on MTG_32004; rounding depends on B unless `MT_XDIAR_ROWS` pins it | ~5 + 0.9 x B ms (7x cheaper per chunk at B=26) |

`MT_XDIAR=1` stays off in the preset until a 30-file cpWER/DER pass decides whether the flip is a draw
or a bias.

## Measured (one RTX-PRO-6000, k6 in-cluster generator, 300 s streams, `xdiar=1` experiment build)

| N | xsession lag p50 / p95 | xsession tick med / p99 | lock path (committed), sweep fork's clean hour-long arms |
|---|---|---|---|
| 20 | 0.09 / 0.19 s | 13.4 / 16 ms | |
| 40 | 0.09 / 0.19 s | 13.6 / 16 ms | |
| 60 | 0.09 / 0.20 s | 14.1 / 18 ms | holds: p95 0.34-0.36 s at 60 and 64, lock 69-74 % |
| 100 | 0.11 / 0.21 s | 19.5 / 30 ms | 80 collapses (lock 97 %) |
| 160 | 0.11 / 0.22 s | 22.4 / 52 ms | |

(My own lock-path control arm collapsed at 60, but it ran `MT_PROFILE=1` with `MT_PROFILE_EVERY=50` -- a
torch.profiler snapshot every 50th step -- which the stepper path never executes; those control rows are a
profiler artifact and are not used here.)

With `xdiar=0` the diarizer costs 3.8 ms per row per tick, so a 16-row tick is ~100 ms (~6 ms per session-chunk).
Hour-long ladder of the shipped configuration (`SWEEP_XSESSION.md`): **60 / 80 / 100 / 160 hold the full hour**
(window p95 0.21-0.37 s at 60-100, <= 0.70 s at 160 with the stepper 94 % busy), **200 collapse**.
`predict_concurrency: 128` (2x the lock path's 64) from that ladder.

## Where the stepper's time goes (2026-09-17, `xdiar=1` slot tables + mel in the stepper, N=160 live)

`{"stats":1}` mid-arm on `nemotron-mt-td-x1` (q406229w/wxe1zly), 160 hour-long real-time streams: stepper busy
**0.55** (the previous build: 0.94 at the same load), tick p50 18.1 / p99 29.3 ms, B p50 2–3 (the stepper keeps
up, so batches stay small). Per tick: ASR 8.3 ms (encoder graph at the 8/16 bucket + label-looping decoder +
per-row hypothesis bookkeeping), fused diarizer 4.8, mel 3.5, gather 0.36, seglst 0.22, cache write-back 0.16.
The diarizer term used to be 3.8 ms x B; it is now ~5 ms flat. What remains per row is ASR-side Python (NeMo's
`merge_to_batched_state` / `batched_hyps_to_hypotheses` / `merge_` / `update_asr_state`, ~0.5 ms per
(session, speaker) row) plus the connection side's share of the GIL (websocket receive + base64 + numpy per
100 ms frame, the partial string join, `mt_turns.pull`).

## Design: stepper in its own process (not built; measured need first)

If the GIL is still the ceiling after the levers above, the remaining move is to give the stepper its own
interpreter: the connection side keeps the websocket, base64, ring buffers and turn building; a child
process (`torch.multiprocessing`, spawn) owns both models and the GPU.

- **PCM in**: one shared-memory float32 ring per slot (2 s of audio, 128 KB; 320 slots = 41 MB) plus two
  shared int64 arrays `write_head[slot]` / `consumed[slot]`. A connection appends its frame into its ring
  (numpy, no IPC per frame); the stepper finds ready rows itself by comparing the heads each tick — zero
  per-chunk messages. The mel window is sliced straight out of the ring on the stepper side.
- **Results out**: per slot a small shared ring of per-step records `(step, offset, per speaker: n_tokens,
  token ids, emission frames)` — the three ints `mt_turns.install` records today plus the new tail of
  `hyp.y_sequence` / `hyp.timestamp` (both already CPU tensors after NeMo's decode). One pipe message per
  tick lists the slots completed; a reader thread in the main process sets the sessions' asyncio events.
- **Turn building** moves to the main process: `SessionTurns.pull` reads the token/frame tail from the
  result ring instead of `ASRState`; it needs only the SentencePiece model (loaded once from the `.nemo`),
  which `IncrementalDetok` already isolates. NeMo's own seglst (the `MT_TURN_SEGMENTS=0` output and the
  `MT_TURNS_VERIFY` reference) stays in the stepper process and is shipped once at the final.
- **Control**: admit(slot, max_speakers) / commit(slot) / release(slot) over the pipe; the stepper zeroes
  the slot's tables on admit exactly as `XStepper.admit` does now.
- **Identity**: the stepper runs the same tick, so the FINAL transcript is unchanged; the turn builder's
  input is the same (ids, frames, offsets) so `segments`/`partial` are unchanged — checked with the 3-file
  cpWER + `MT_TURNS_VERIFY`.
- **What it buys**: every connection-side Python leaves the stepper's interpreter (and vice versa) — two
  cores of Python instead of one. Cost: ~500 lines, one more process to supervise, and the stepper's own
  Python per row is untouched, so it only helps when the connection side is what starves the stepper.

## Sending is decoupled from receiving

Each connection has a sender task draining a one-slot mailbox. The first hour-long N=60 ladder run showed
lag p50 0.56 / p95 5.5 s with an idle stepper (queue wait 0.2 ms, tick 13 ms): the handler used to serve a
connection sequentially (receive -> step -> `await send_text`), so a slow reader of the growing replace-style
partials (6 -> 89 KB over an hour, ~0.5 MB/s per connection on the path) stalled that connection's next
audio frames and its chunks reached the stepper late. Now an unsent partial is superseded by the next one
(replace semantics), finals / profile / error frames are always delivered, and the per-session coalesced
count is logged. Rerun: steady-state windows p95 0.27 s at N=60 (was 3.7 s).

## Knobs

- `MT_XSESSION` (1): stepper on; 0 = the per-session lock path.
- `MT_XSESSION_B` (16): max sessions per tick. `MT_XSESSION_ROWS` (32) / `MT_XSESSION_ROW_BUCKETS`
  (8,16,32): padded ASR speaker rows; one encoder graph per bucket, warmed at load; more rows = slabs.
- `MT_XSESSION_WAIT_MS` (0): coalescing window after the first ready session.
- `MT_XDIAR` (0) / `MT_XDIAR_ROWS` (0): diarizer mode / pinned diarizer batch.
- `MT_WORKERS` must exceed `predict_concurrency` with margin: each live session parks a pool thread
  while it waits for its tick, and `message()` runs on the pool too.
- Runtime (`MT_ALLOW_CONTROL=1`, handshake `{"control": {...}}`): `xsession_wait_ms`, `xsession_b`,
  `xdiar_rows` flip under traffic; `xdiar`, `row_buckets` only with no live session.

## Validation of the registry layout (fde-internal, `nemotron-mt-td-registry-xs` wozor9g3/qrm9760)

- Load (validation build, still with a 64-row bucket): `ready in 34s`, 4 encoder graphs, 133 sync-mode diarizer graphs captured by the
  180 s warm session through the stepper, decoder `full_graph`, 3.7 GB allocated / 8.3 GB reserved.
- Load fault found, 64-row bucket dropped: the first 64-row ASR call of the process (encoder graph capture +
  decoder state at that width) hit `cudaErrorIllegalAddress` on every load of the registry layout -- as the
  first bucket or after 8/16/32, with 8 active + 56 pad rows or 64 active, on white noise or silence (the
  experiment build had loaded at 64 in 1 of 3 tries, the validation build on its 3rd retry); 8/16/32 never
  failed. Shipped: `MT_XSESSION_ROWS=32`, buckets 8/16/32, slabs above. Root cause (NeMo label-looping
  decoder or encoder graph wrapper at batch 64 on nemo:26.08) not chased; on the day-0 list.
- 3-file FINAL cpWER (canonical scorer): **23.57 / 28.38 / 18.45, macro 23.47** — identical to the committed
  build including ins/del/sub (25/197/116, 35/251/113, 34/149/91).
- First connection 11 s after ACTIVE (60 s of MTG_32000, real time): lag p50 0.07 / p95 0.14 / max 0.17 s,
  no backlog; the clip's cpWER 23.32.
- Malformed frames (`rsi-bench/nemotron_diar/mt_probe_malformed.py`): non-JSON, non-object, bad base64, odd
  PCM byte count, `max_speakers=99` -> each an `{"type":"error"}` frame + clean close in ~0.5 s; a normal
  session afterwards completes (`processed_s` 3.36 for 3 s of audio).
