# Nemotron 3 Diarization (batch)

> Batch HTTP preset for NVIDIA's [`nvidia/Nemotron-3-Diarization`](https://huggingface.co/nvidia/Nemotron-3-Diarization)
> (General Access checkpoint), released under the [OpenMDW 1.1 license](https://openmdw.ai/license/1-1/)
> (commercial use permitted). The `.nemo` is downloaded at deploy from that repo; it is not committed here.
> Sibling presets: `../streaming` (live diarization over a WebSocket) and
> `../../nemotron-3-diarized-transcription/streaming` (speaker-tagged words).

## What it is
NeMo **streaming Sortformer** speaker diarization ("who spoke when"), 100M params,
31-layer Transformer encoder (RoPE), 80 ms encoder frame rate. Handles **up to 8
speakers**, labels ordered by arrival time. A single checkpoint supports latency
profiles from 80 ms to a 30.4 s offline-style buffer; chunked inference has no max
duration. See the HF model card / `overview.md` for full details.

## Truss shape
- **Base image:** `nvcr.io/nvidia/nemo:26.08` (contemporaneous NeMo Framework container;
  ships the streaming `SortformerEncLabelModel` + matching torch/CUDA).
- **Weights:** `.nemo` (189 MB, bf16) mounted at `/models/nemotron-diar` from
  `hf://nvidia/Nemotron-3-Diarization` (public, no token needed).
- **Model code:** `model/model.py` restores the checkpoint to CUDA, applies a streaming
  profile, ffmpeg-normalizes input to 16 kHz mono, runs `diarize()`, and returns turns.
- **Hardware:** RTX-PRO-6000 (a supported Blackwell SKU; matches our community-1 anchor box).

## API
Input:
```json
{"diarization_input": {"audio": {"url": "..."}, "latency": "offline"}}
```
`audio` accepts `url` or `audio_b64`. `latency` ∈ `offline` (default, 30.4 s, best DER),
`low` (1.04 s), `ultralow` (0.32 s) — the model card's 80 ms-frame profiles. `verylow` (0.64 s)
is measured in `BENCHMARK.md` but not offered by default (PyTorch Inductor cannot compile its
chunk shape, so it would run eager); opt in with `NEMO_DIAR_PROFILES=offline,low,verylow,ultralow`. Output:
```json
{"turns": [{"start": 0.51, "end": 12.62, "speaker": "speaker_0"}, ...],
 "segments": ["..."], "num_speakers": 3, "latency": "offline", "batch_n": 1,
 "timing": {"fetch_ms": 14, "decode_ms": 169, "queue_ms": 181, "batch_n": 1, "batch_ms": 83,
            "fwd_ms": 67, "post_ms": 11, "audio_s": 363.38, "total_ms": 278}}
```
`batch_n` is how many requests shared the model call; `timing` is the server-side profile of this request.

## Performance
The model runs NVIDIA's whole-file streaming loop, but every step — the Inductor-compiled encoder,
the head, and NeMo's speaker-cache/FIFO update — is replayed from a **CUDA graph captured per state
shape** (`model/graph_runner.py`; 129 shapes at `low`, 237 at `ultralow`, 2 at `offline`, × 14 / 8 / 8
captured batch sizes — every 2 rows from 8 at `low`, every 4 elsewhere, none below 4 — all captured at load). Against NVIDIA's reference script (`e2e_diarize_speech.py`,
`compile_encoder=true`, bf16) on the same RTX PRO 6000 — reference measured on the preview checkpoint,
ours re-measured on the GA checkpoint (its `low` profile carries a larger per-step state: chunk 9 + rc 4,
speaker cache 264): `low` 8 six-minute files in one batch **2.01 s vs 2.66 s** (RTFx **1,444** vs 1,091;
NVIDIA's best cell, bs=32, is 1,273; preview build 1,478), a single file **1.28 s vs 2.13 s** (preview 0.85 s);
`ultralow` 8 six-minute files **6.13 s vs NVIDIA's 33.9 s / 32 files at bs=8** (RTFx 474 vs 348), a single
file 4.02 s (RTFx 90); `offline` bs=8 0.11 s (~26,000 RTFx), a single file 68 ms (5,346), parity with the
reference. Sustained open-loop throughput at `low`: **200 six-minute files per minute held** (ceiling
≈ 207–210/min; 240/min backlogs), unchanged from the preview build.
`low` runs a **hybrid attention** (`NEMO_DIAR_ATTN_MODE_LOW=hybrid`): flash SDPA on every step where
all active rows have a full chunk — exact in NeMo's sync state, where every row shares the cache/FIFO
fill — and NeMo's masked FlexAttention only on a row's final partial chunk (−4…−12 % step time, drift
class unchanged); the TF32 profiles keep FlexAttention.
Full ladder, model-card table and the DER gate with a NeMo-eager control arm in `BENCHMARK.md` →
Performance. On top of that the server:
- **coalesces independent HTTP requests** of the same profile into one batched call: a request enqueues
  on receipt and decodes (`ffmpeg` on pipes into pinned memory) while the window runs; the batch starts
  after 100 ms of quiet once every admitted decode is done, longest files first
  (`NEMO_DIAR_MB_WINDOW_MS`=100/`_MAX_WAIT_MS`=1500/`_CAP`=32); a collector thread forms the next
  batch while the GPU runs the current one and holds the window open until the forward finishes, so
  the GPU runs back to back (N=64 closed loop: 171 files/min, 0.96× NVIDIA's lockstep ceiling with
  upload and decode in the loop).
- runs NeMo's post-processing once on the GPU for the whole batch (identical segments; NeMo's per-file
  CPU version costs more than the forward at `offline`).
- returns a **per-request timing profile** (`timing`: fetch, decode, queue wait, batch size, graph batch
  size, h2d, mel, steps, forward, post-processing, total) and logs one line per batch with RTFx.
Precision: **bf16 on every profile** (`NEMO_DIAR_DTYPE`, NVIDIA's card precision) — the rule is bf16 wherever
it is at least the vendor's compiled-bf16 quality on NOTSOFAR-129 vs NeMo eager: `low` set-level within ±0.1
at batch ≥ 8; `ultralow` +0.16…+0.37 across four runs (NVIDIA's compiled path: +0.25 at bs=32, +0.42 at K≈8 — comparable,
not clearly better; per-file 0.8 vs 0.9); `offline` −0.01 / −0.09 at K≈8 / K≈32 (NVIDIA's path +0.15). **fp32 (TF32 matmuls) is the strict, eager-quality knob** (`NEMO_DIAR_DTYPE_<PROFILE>=fp32`):
`ultralow` +0.05 set-level, mean per-file |ΔDER| 0.20, worst file 1.4 (full gate pass) at 0.46× the bf16
speed; `offline` +0.01 / 0.105 / 1.8 at 0.52×; `low` 0.32 vs NeMo's own batched eager 0.22 at 0.61×.
**No batch-size-1 graph is ever run**: dynamo's batch-1 specialisation of the compiled encoder under-detects
speech by +0.7–1.0 pt in bf16 and TF32 alike (NVIDIA's compiled script has the same bias); a single request
runs as identical rows at the same latency. Every response carries
`nonfinite_preds` / `nonfinite_state` / `pred_max` in `timing` (0 / 0 / 1.0 across 129 files and a
64-minute stress file for every core). `NEMO_DIAR_ENGINE=nemo` keeps the reference-script path (torch.compile,
growing state) for A/B; `NEMO_DIAR_COMPILE=0` replays eager kernels (slower at bs ≥ 8).
Startup: restore + compile + capture ≈ 2–3 min per profile (`startup_threshold_seconds` 1200);
b10cache keeps the Inductor artifacts for scale-ups. The NeMo overlay is pinned to commit
`3c2d62ae7eb4`. For a library listing: min 1 replica, `concurrency_target` ≈ 16, `predict_concurrency` 48.

Client errors — missing `diarization_input.audio`, unknown `latency`, invalid base64, an
unreachable `url`, or audio ffmpeg cannot decode — return **HTTP 400** with the reason; only
model/infra faults return 500.

## Release state (General Access checkpoint, 2026-09)
The GA checkpoint (`nvidia/Nemotron-3-Diarization`, OpenMDW-1.1) replaced the early-access preview
on 2026-09-21: retrained bf16 weights (189 MB, sha256 `867c53f5…`), speaker cache 264 frames, the
same architecture and the same latency profiles this preset already shipped. Everything in the
performance section above was re-measured on it (`rsi-bench/nemotron_diar/release/GA_VALIDATION.md`).
Autoscaling for the listing: min 1 replica, `concurrency_target` ≈ 16.

## Benchmark
See `BENCHMARK.md` for our DER measurements (protocol: pyannote.metrics, collar 0.25 s
and 0.0 s, overlap **included**) against our local VoxConverse / AMI / NOTSOFAR mirrors,
alongside NVIDIA's reported numbers and our community-1 reference.

> **⚠️ Data-contamination caveat.** Per the model card's training tables,
> **VoxConverse v0.3 dev+test**, **AMI train+dev**, and **NOTSOFAR1 train+dev** were in
> the training data. VoxConverse numbers here are therefore *contaminated* (train-on-test).
> The AMI **test** and NOTSOFAR **eval** splits are the honest comparisons; NVIDIA further
> scores AMI/AliMeeting/NOTSOFAR with **forced-alignment** RTTMs, which differ from our
> mirrors' references, so absolute numbers are not directly comparable to the card.