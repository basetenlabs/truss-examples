# Nemotron 3 Diarized Transcription (batch)

> **English-only.** Batch HTTP preset for NVIDIA's coupled overlap-aware speaker-attributed
> transcription: the diarizer [`nvidia/Nemotron-3-Diarization`](https://huggingface.co/nvidia/Nemotron-3-Diarization)
> (General Access checkpoint, [OpenMDW 1.1](https://openmdw.ai/license/1-1/), commercial use permitted)
> drives one ASR instance per speaker of `nvidia/multitalker-parakeet-streaming-0.6b-v1` (NVIDIA Open
> Model License). Both checkpoints are pulled at deploy from Hugging Face, never committed.
> Sibling: `../streaming` (live, over a WebSocket); diarization only: `../../nemotron-3-diarization`.

## What it is
"Who said what", overlap-aware. Nemotron-3-Diarization (streaming Sortformer) emits
per-speaker activity; **multitalker-parakeet-streaming-0.6b-v1** (600M FastConformer-RNNT,
built on Nemotron-Speech-Streaming-en) runs **one ASR instance per speaker** on the *same
mixed audio*, adapting to each speaker by injecting speaker kernels into its pre-encode
layer — no enrollment, no embeddings. Because each instance focuses on one speaker, fully
overlapped speech is transcribed per speaker, which a diarize→cut→transcribe pipeline
cannot do (a cut segment still contains every overlapping voice).

Integration is NVIDIA's own (`SpeakerTaggedASR` in NeMo), run as simulated cache-aware
streaming over the whole file: ASR attention context `[70, 13]` (~1.12 s advance), diarizer
`spkcache 264 / fifo 264 / update 222`, `parallel_speaker_strategy`, `cache_gating`,
`binary_diar_preds`, bf16.

## API
```json
{"transcription_input": {"audio": {"url": "..."}, "max_speakers": 8}}
```
`audio` accepts `url` or `audio_b64` (any ffmpeg-decodable format; normalized to 16 kHz
mono). `max_speakers` (1–8, default 8) caps the per-speaker ASR instances. Response:
```json
{"segments": [{"speaker": "speaker_0", "start": 0.51, "end": 4.2, "text": "..."}, ...],
 "speakers": 3, "text_by_speaker": {"speaker_0": "...", "speaker_1": "..."},
 "compute_s": 21.6, "peak_gpu_gb": 4.6, "batch_n": 1}
```
Speaker labels are session-local and arrival-ordered — not identities. `compute_s` is the
coupled session's compute (shared by the `batch_n` requests that ran in it).

## Measured (see `BENCHMARK.md`)
NOTSOFAR-1 eval, first 30 sessions, cpWER (meeteval, Whisper-normalised): **28.58 macro / 28.89 micro
on the GA diarizer** (paired −1.59 vs the preview diarizer, better on 22/30 sessions; no code change —
`GA_VALIDATION.md` §6). The rest of this section was measured on the preview diarizer: 30.2 macro /
30.5 micro (shipped: segments from the turn builder; NeMo's own seglst output scored 31.9 / 32.2
because it re-appends a speaker's whole transcript on a punctuation prefix break — see *Output path*
in `BENCHMARK.md`; the first bf16 measurement was 31.1 / 31.7); full 129 sessions under the seglst
output **28.9 / 30.3**. **Matched** against our stack on the same 30: Qwen3-ASR-1.7B + Nemotron-3
offline turns 31.8 / 32.3 (a tie — the edge is <1 pt, only on 4-speaker meetings); Qwen3-ASR +
pyannote turns 44–45 (the diarizer choice dominates).

Cost on RTX-PRO-6000, one replica (`MT_FAST=1`, the streaming preset's CUDA-graph step dispatch —
`packages/mt_fast.py` — bf16 ASR pinned at 32 rows, the diarizer in 8-session slabs, cap 16): a
16-session coupled batch of ~6-min files computes in **14.5 s**. Measured from an in-cluster load
generator (never a laptop), cpWER scored from the same runs:

| arm | sessions/min/replica | latency p50 | cpWER macro / micro at load |
|---|---|---|---|
| closed-loop K=30 | **61.0** (e98c7e8 build: 15.9 at K=16) | 28.8 s, flat | 31.90 / 32.21 (seglst output; turn builder: 30.17 / 30.46) |
| open-loop 50/min | **49.3, holds** (flat latency 13.8 s) | 13.8 s | 31.90 |
| fp32 ASR (`MT_ASR_DTYPE=fp32`), cap 16 | 41.7 closed / 40 open holds | 42 s / 28 s | 31.90 / 32.23 |

**Reproducibility.** bf16 output depends on the ASR row bucket: between 8- and 32-row pinning, 0/30
eval-30 files are byte-identical (mean |Δ| 0.32, max 2.06 cpWER), with or without cuBLAS's reduced-
precision reductions. At one pinned shape it is deterministic (closed-loop vs open-loop runs: 30/30
files byte-identical), and at the serving shape bf16 vs fp32 is mean |Δ| 0.37 / max 1.77 with equal
macro (31.899). fp32 (`MT_ASR_DTYPE=fp32`) is shape-invariant (M=8 == M=32; 29/30 files identical
across client concurrencies, the 30th is the bf16 diarizer's row-position residual) and restores
reproducibility across buckets at ~1.4× the step (42 vs 61 sessions/min). Full ladder, the attribution
and the rejected cap-16-with-16-row-diarizer draw in `BENCHMARK.md`.

## Truss shape
Same container recipe as `stt/nemotron-3-diarization` (`nvcr.io/nvidia/nemo:26.08`
+ `[asr]` deps + NeMo main overlay; `multispk_transcribe_utils` lives in NeMo main). Both
models are loaded once. A single consumer thread collects requests for `MT_MB_WINDOW_MS`
(100 ms, plus whatever queues while a session runs), groups them by `max_speakers` (it binds
per session) and runs ONE coupled session with up to `MT_MB_CAP` (16) files on NeMo's batch
dimension — all rows start at step 0 together, shorter files zero-padded to the longest and
their output cut back at the true duration (`MT_MB_PAD_WASTE` 0.3 splits a window when padding
would waste more than that fraction of rows). `predict_concurrency` 48 must exceed the cap (16) so
a full batch can form. Nemotron's FeatureStacking pre-encoder requires
`pad_and_drop_preencoded=True`; the diarizer's chunk is set to the ASR's per-step output span
so activity and audio stay aligned.

**Numerical determinism.** bf16 GEMM rounding on this GPU depends on the batch shape (and on
the row's tile within a call), and the 24-layer streaming conformer + greedy RNNT amplify one
ulp into different words — so an unpinned B=8 session never bit-matches B=1 and per-session
cpWER swings ±10–20. `packages/multitalker_batch_fix.py` runs the diarizer in slabs of `MT_DIAR_ROWS`
(8) sessions so every diarizer call sees the 8-row shapes (a 16-row call is a different draw, +0.56);
`mt_fast.FastPath` runs the ASR as slabs of `MT_PAD_ROWS` (32) speaker rows — one encoder CUDA graph
at that shape — in bf16 (`MT_ASR_DTYPE`; fp32 without autocast is the shape-invariant alternative,
M=8 == M=32 byte-identical, ~1.4× the step). Result: B=1…16 give the same bytes per file (30/30 eval
sessions identical between cap 8 and cap 16), and repeated runs are identical. Changing cap,
`MT_DIAR_ROWS`, `MT_PAD_ROWS` or dtype changes the numerics — re-benchmark. `MT_FAST=0` restores the
eager e98c7e8 step (then `MT_ASR_ROWS` / `MT_ASR_FP32` pin the ASR).

## Caveats
- **English-only** ASR.
- The commercially licensed `diar_streaming_sortformer_4spk-v2.1` pairing was measured and
  **is not competitive** (cpWER 47.2) — purely its 4-speaker cap — and in bf16 it hits a
  cuBLAS error that poisons the CUDA context. This truss ships the Nemotron pairing only.
- On aggregate cpWER this pipeline ties our Qwen3-ASR + Nemotron-turns stack; its distinct value
  is transcribing fully overlapped speech and a single-model streaming path (see `../streaming`).
- Batch only (whole file); the live WebSocket variant is `../streaming`.