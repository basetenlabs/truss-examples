# NVIDIA Nemotron-3.5-Transcribe-30B-A4B — Unified Real-time

Nemotron 3.5 Transcribe over [baseten-stt](https://github.com/basetenlabs/baseten-stt)'s
unified WebSocket protocol (`/v1/realtime`). The model itself is batch-only; baseten-stt
turns it into a real-time endpoint the same way it serves Whisper and Qwen3-ASR:

- **Turns** end on `client_flush` (the client sends `input_audio_buffer.flush`) or
  `server_vad` (the server's Silero VAD ends the turn on silence).
- **Finals** decode the whole turn once.
- **Partials** (opt-in per session) re-decode the trailing 8 s of the turn on the session's
  negotiated interval. They are `replace`-style: each supersedes the last.

Listed in the Model Library as "NVIDIA Nemotron 3.5 Transcribe 30B-A4B (Real-time)", next to
the batch preset. Serves NVIDIA's public checkpoint, as the batch preset does.

## How it is built

The Qwen3-ASR unified preset (`stt/qwen3-asr-1.7b-streaming/unified`) loads vLLM in-process.
That doesn't work here: the model needs NeMo Speech's vLLM plugin on vLLM 0.28 with a CUDA 13
torch stack (NVIDIA validates Python 3.13), while baseten-stt runs on the server image's Python
3.12 with its own pins. So:

| Piece | What |
|---|---|
| Server | `baseten/stt-server:0.2.0` (baseten-stt v0.2.0), Python 3.12 |
| Backend | `packages/nemotron_stt`: the template's stream unchanged except for the name, and an engine that calls a sidecar |
| Engine | The batch preset's `vllm serve` (`stt/nemotron-3.5-transcribe-30b-a4b/rtx-pro-6000/batch`, NVIDIA's model-card flags: MTP-8, FP32 SSM state, utilization 0.94, 65,536-token window, no async scheduling; thinking off), launched by `build()` from `/opt/nemo-speech/venv` on `127.0.0.1:8000` |
| Decode | One `/v1/chat/completions` request per final or partial: the turn's PCM as WAV, prompt "Transcribe this audio." unless the session sets a language or context (below) |
| Cancel | Cancelling a decode closes its request; vLLM aborts requests whose client disconnected |
| CUDA toolchain | The slim image has no CUDA toolkit, but FlashInfer JIT-compiles its SM120 fused-MoE kernels at first start, so the build installs nvcc, crt, nvvm and nvrtc 13.0 wheels into the venv and the sidecar points `CUDA_HOME` at them |

Weights, build and engine envelope match the batch preset (see its README):

- **Weights.** The public `nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B` at revision
  `a188fff`, mounted at `/app/checkpoint/model` (`BASETEN_MODEL_PATH`). No secret is needed.
  The public `config.json` renamed the architecture, so the sidecar passes the batch preset's
  `--hf-overrides` with the name the pinned Speech commit registers.
- **Build.** The model card's locked install, with the batch preset's pins. The one difference
  is the CUDA toolchain from wheels above, because this image has no `/usr/local/cuda`.
- **Envelope.** The batch preset's `vllm serve` flags, written out in
  `packages/sidecar/start_vllm.sh`, and its `environment_variables`. The sidecar drops only
  the CTC timestamp middleware and listens on localhost.
- **Turn length.** stt-server 0.2.0 cuts a turn at 29 s (`BASETEN_STT_MAX_TURN_S`, unset here),
  as for the other unified presets, and the final says `ended_by: max_turn_length`. That keeps
  a final's decode bounded. The model's 65,536-token window would hold about an hour per turn,
  so the cap can be raised (up to 3600, or `0` for none) at the cost of final latency.

No probe answers until the sidecar is ready and warm. A new deployment measured ~10 min from
sidecar start to ready: weights, ~300 s of engine init, and the warmup. That needs the SM120
fused-MoE kernel compiled into the image (`config.yaml`, ~5 min of the build); compiled by
FlashInfer at first start instead, it ran 25-45 min silently after "Warming up Mamba2" on every
new deployment, and FlashInfer's prebuilt wheel of it stalled ~3% of finals for 2-8 s. `build()`
then decodes 1-29 s turns of seeded noise, one at a time and then 32 at once, so the first
decodes' Triton compiles happen before the replica reports ready (`NEMOTRON_WARMUP=0` skips it).

After startup a thread watches the sidecar. baseten-stt's probes only check that the backend was
built, so if vLLM exits the thread stops the server (SIGTERM, then a hard exit after 30 s) and the
platform replaces the replica, instead of keeping one that fails every decode.

A decode that stops on `max_tokens` (12 tokens per second of audio, about 4x the model's rate) is
a truncated transcript, so the turn fails with `transcription.failed` and the session continues.
A partial tick whose audio hasn't changed since the last partial is skipped.

### Language and context

The model takes both as instructions, so the backend writes them into the decode's prompt
(`engine.prompt_for`). A client sets them with `session.update`, and every decode in the
session, partials included, carries them:

| Session sets | Prompt |
|---|---|
| Nothing (`language` defaults to `auto`) | `Transcribe this audio.`, as measured below |
| `language: "en"` (any of the card's 40, as BCP-47 codes) | The card's verbatim ASR instruction with the language named: `Audio transcription. Language: English. Wording: verbatim, including fillers and repetitions. Formatting: punctuation and capitalization. Output: transcript only.` |
| `context.text` | Appended to the instruction as written: domain framing, or a further instruction such as the card's near-field one (`Transcribe only the speaker nearest to the microphone. Do not transcribe any other voices.`) |
| `context.keyterms` | Appended in the card's contextual-ASR form: `Context terms: <term>, <term>.` |

The context slot holds 256 words across text and terms.

#### On noisy telephone audio

16 τ-voice retail calls (111 min of the caller's side: 8 kHz μ-law with street, metro, crowd
and TV noise), replayed through a voice-agent chain into this preset, every session with the
same turns from one Silero VAD pass. Scored against what the simulated caller said:

| Session sets | Caller words wrong or missed | of which missed | Words added | WER | Digit runs caught | Final after VAD stop, p50 |
|---|---|---|---|---|---|---|
| Nothing (`auto`) | 26.5% | 14.0% | 20.5% | 47.0% | 75/104 | 0.53 s |
| `language: "en"` | 26.1% | 14.9% | 16.7% | 42.8% | 77/104 | 0.53 s |
| `en` + the card's near-field `context.text` | 34.7% | 24.3% | 4.0% | 38.7% | 58/104 | 0.51 s |
| *Nemotron ASR NIM, for reference* | *24.7%* | *16.0%* | *2.2%* | *26.9%* | *76/104* | *~0 s* |

Most added words are background voices transcribed as the caller's. With `auto` some come out
in their own language -- in a full 114-call τ-voice run, 717 of 4,296 finals were Tamil,
Russian or Chinese -- and on noise-only turns the model sometimes answers instead of
transcribing ("I'm sorry, I cannot provide a transcription...": 15 finals here with `auto`,
3 with `en`, 0 with near-field).

- **`language`:** name it when it's known. `en` removed most of the added words at no cost to
  the caller's.
- **Near-field:** keeps other voices out, but also drops parts of the caller on noisy phone
  audio, a third of the digit runs among them. Not a default for phone agents.
- **Key terms:** not measured here yet.

## Results

### Public checkpoint (PR CI)

The PR's b10-bench run (`run-851f331762a3`, `workload.json`'s cells, 1x RTX PRO 6000, partials
on) on the public checkpoint at `a188fff`, served by stt-server 0.2.0. 0 errors.

| | c1 | c32 |
|---|---|---|
| Final latency p50 / p90 | 1,267 / 1,396 ms | 1,493 / 1,885 ms |
| Time to first partial p50 | -197 ms | 799 ms |
| Throughput | 0.09 streams/s | 2.07 streams/s |

| WER (naive, 150 samples, concurrency 8) | Final | Partial |
|---|---|---|
| pipecat | 1.44% | 20.73% |
| mu-bench-full | 22.61% | 27.31% |

The c1 p99 (50 s) is the one stalled stream per run noted under the v1.0.1 results below.

### Earlier runs on v1.0.1

Everything below was measured on v1.0.1 with the same model-card flags, downloaded from NGC,
before this preset moved to the public checkpoint. (The v1.0.1 deployments passed
`--mamba-cache-mode all`; with prefix caching off vLLM 0.28.0 resolves that to `none`, as this
preset now states.)

#### v1.0.1, model card recipe

b10-bench (`baseten_stt_unified`, pipecat, partials every 0.5 s, 1x real time, 0.5 s server
VAD), deployment `wl5vzd03` / `w60m272` on 1x RTX PRO 6000. 528 streams in all, 0 failed.
Final latency is from the end of the turn's audio to its final transcript.

| Cell | Result |
|---|---|
| Quality, `client_flush`, 100 clips, concurrency 8 | naive WER 1.27% |
| Quality, `server_vad`, 100 clips, concurrency 8 | naive WER 1.61% |
| Latency, `client_flush`, 64 clips, concurrency 1 / 8 / 16 / 32 | final p50 1.26 / 1.36 / 1.40 / 1.51 s |
| Latency, `server_vad`, 64 clips, concurrency 1 / 8 / 16 / 32 | final p50 0.76 / 0.88 / 0.83 / 0.93 s; p99 at 8-32 2.1-2.5 s |

On the EA checkpoint (MTP-4 + async, 0.2 s server VAD) the same set measured 2.24%
(`client_flush`) and 6.56% (`server_vad`) WER, and `client_flush` final p50 1.24 -> 1.64 s.

- **First sessions after boot returned nothing** in these runs. The first decodes JIT-compile
  Triton kernels (vLLM logs `eagle_*` and `_causal_conv1d_fwd_kernel` compiles), the client
  closes, and the server's 5 s drain gives up. The startup warmup now runs those decodes before
  the replica reports ready.
- **One concurrency-1 stream per run stalls ~55 s.** It happened in both modes here, and once
  (57 s) on the EA checkpoint. It is not a kernel compile (no JIT warnings in that window), and
  it doesn't recur at concurrency 8-32, which replay the same 64 clips. Cause not found.

## Capacity

On v1.0.1, one replica sustains about **32 real-time streams** with partials every 0.5 s. Past that, latency
grows by tens of seconds while completed streams level off at ~2.3-3.1/s, so `predict_concurrency`
stays at 32 and traffic beyond it should scale out.

High-concurrency sweeps (b10-bench, 128-256 pipecat clips per level, caps lifted to 256
connections / 288 sessions so the server, not admission, sets the limit). Final latency p50:

| Streams | RTX PRO 6000, MTP-8 (`31rp519`) | H100, MTP-8 (`q92n585`) | RTX PRO 6000, MTP-2 + async (`wdxjlrk`) |
|---|---|---|---|
| 32 | **1.65 s / 0.98 s** | 1.82 s / 1.27 s | 2.23 s / 1.42 s |
| 64 | 10.8 s / 7.3 s | 8.1 s / 8.2 s | 7.9 s / 7.9 s |
| 128 | 25.1 s / 21.0 s | 25.0 s / 15.6 s | 18.8 s / 19.4 s |
| 256 | 32.2 s / 14.2 s (72% `server_vad` streams failed) | 26.1 s / 27.1 s | 45.4 s / 24.5 s |
| Completed streams/s, 64-128 | 2.8-3.1 | 2.4-3.0 | 2.3-3.1 |

Each cell is `client_flush` / `server_vad`. Runs:
[run-acda244e0d09](https://batch-bench.internal.basetensors.com/?view=runs&run=run-acda244e0d09),
[run-e3710f134eea](https://batch-bench.internal.basetensors.com/?view=runs&run=run-e3710f134eea),
[run-a03f0db0ccaf](https://batch-bench.internal.basetensors.com/?view=runs&run=run-a03f0db0ccaf).
With the shipped caps (32 connections, 48 sessions;
[run-8d6c1644e7ad](https://batch-bench.internal.basetensors.com/?view=runs&run=run-8d6c1644e7ad)),
latency stays at 1.6 s / 1.0 s through 48 streams, and streams past the cap fail instead of
queueing: 5-15% at 64 and ~50% at 128.

- **H100 doesn't raise the ceiling.** The 64 GB checkpoint leaves 7.6 GiB of KV cache at 0.94
  utilization (vLLM: "maximum concurrency for 65,536 tokens per request: 7.88x"), so vLLM runs
  7-13 requests at once and queues up to ~200.
- **MTP-2 + async scheduling doesn't either.** It removed vLLM's queue (up to 43 running, 0
  waiting, KV cache at most 35%) without changing throughput, and is slower at 32 streams.
- **The limit is upstream of the vLLM engine.** GPUs were ~50-65% busy and the container used
  ~2.5 of its 15 CPU cores with no throttling, which points at a single-threaded process
  (baseten-stt's server or vLLM's single API server) rather than the GPU. Not yet confirmed with
  per-process CPU.

## Compared with Nemotron ASR Streaming

The motivation for these sweeps: whether Nemotron Transcribe can replace
[Nemotron ASR Streaming](../../nemotron-asr-streaming-0.6b/streaming-multilingual/README.md) (0.6B
cache-aware RNNT, NIM 1.4.0) as a voice agent's STT. Its numbers are from that preset's
2026-10-09 sweep, not rerun here.

| 1x H100 | Nemotron Transcribe (this preset, v1.0.1) | Nemotron ASR Streaming |
|---|---|---|
| Model | 30B MoE speech LLM (~4.3B active) | 0.6B streaming RNNT |
| Streams within real time | ~32 | ~416 |
| Final latency at that load (p50) | 1.3-1.8 s | 0.51 s (p95 0.75 s) |
| At 128 streams | p50 15.6-25.0 s | p50 0.32 s, p95 0.34 s |
| At 256 streams | p50 26-27 s | p50 0.43 s, p95 0.56 s |
| Where it saturates | between 32 and 64 streams | ~450 streams (4.6 s at 512) |
| Limiting factor | per-request work upstream of vLLM; KV cache on H100 | NIM's WebSocket bridge (gRPC holds 0.8 s at 1000) |
| WER, pipecat streaming | 1.3% (`client_flush`) / 1.6% (`server_vad`) | not measured on these clips |

Per GPU, Nemotron ASR Streaming serves roughly 13x the concurrent streams at a third to a fifth
of the latency, so replacing it with Transcribe means roughly 13x the GPUs for the same traffic,
in exchange for Transcribe's accuracy. Read the ratio as an order of magnitude: the harnesses,
audio (short pipecat turns here, ~65 s streams there) and latency definitions differ.

Levers not yet measured: partials every 1-2 s instead of 0.5 s (fewer decodes per stream), the
FP8 checkpoint (more KV cache), and more server processes (`--api-server-count`, baseten-stt
workers) if the single-process hypothesis holds.

## Tests

```bash
PYTHONPATH=stt/nemotron-3.5-transcribe-30b-a4b-streaming/unified/packages \
  uv run --no-project --python 3.12 --with pytest --with pytest-asyncio --with httpx \
  --with <path to baseten-stt> pytest stt/nemotron-3.5-transcribe-30b-a4b-streaming/unified/tests -q
```

The stream tests are the template's. The engine tests cover the request body, response
parsing, the empty-output and truncated cases, cancellation, sidecar startup failure, the
sidecar supervisor and the warmup audio. One stream test covers skipping unchanged partials.