# VibeVoice-ASR-Streaming-7B

[Microsoft VibeVoice-ASR-Streaming-7B](https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-7B)
is a streaming, speaker-attributed ASR model: it transcribes **who said what** as
speech arrives, with no separate diarization stage, across 10 languages
(en, zh, es, pt, de, ja, ko, fr, ru, it). MIT licensed. ~8.7B parameters — a
Qwen2.5-7B decoder (28 layers, hidden 3584) plus VibeVoice's continuous
acoustic and semantic audio tokenizers, ~17.35 GB at bf16.

It is served through [Microsoft's own vLLM plugin](https://github.com/microsoft/VibeVoice),
pinned to commit `1541f590c70`, on `vllm/vllm-openai:v0.14.1` — the image
[upstream's deploy doc pins](https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-vllm-asr-streaming.md).
The plugin registers this checkpoint's architecture
(`VibeVoiceForASRStreamingTraining`) with vLLM through the
`vllm.general_plugins` entry point, so no vLLM source change is needed.

Weights are pinned to commit
[`60d858b5`](https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-7B/tree/60d858b518b4e19d404af3737f848fc185b30177).

## Read this before trusting a green bench

Two things this package's CI results do **not** tell you.

**1. The streaming route is measured now — but its numbers are paced, and the
chat route's are not.** The declared protocol is `qwen_realtime`, a streaming
contract, so CI submits `perf.stt_streaming` + `quality.stt_streaming` and
measures TTFP, finalization lag and real-time factor over a live WebSocket.
**Seven bench runs are on record**, all tabulated on
[PR #395](https://github.com/basetenlabs/model-registry/pull/395). The first
two (`run-ddaef0464259`, `run-9bf9ba29d292`) came from the previous
`openai_chat_audio` contract against `POST /v1/chat/completions` — a whole clip
in, a whole transcript out. Runs 3-6 are the streaming route on the H100 arm
that was benched and then removed (see
[Accelerator](#accelerator-rtx-pro-6000-chosen-over-h100-on-measured-cost)),
three of them repeats of one identical config. Run 7 (`run-abcda730b5ac`, GHA
run [33797138429](https://github.com/basetenlabs/model-registry/actions/runs/33797138429),
endpoint `wss://…/deployment/woompkv/websocket`) is this preset's own card,
green with 0.00% request errors at every level of c ∈ {1, 4, 16}. So the
streaming path **is** measured for perf and quality, on the hardware that
ships. What the two sets of numbers are
**not** is one series: a streaming protocol **paces audio at real time**, while
the chat route and the `stt/parakeet-tdt-0.6b-v3` baseline submit whole clips as
fast as they are accepted. A paced per-stream figure and an unpaced
clip-throughput figure measure different things — so do not compare them and do
not put them in one table; any ratio between the two is a category error. The
exposure mechanism and the wire translation are reviewed under
[The streaming route](#the-streaming-route), which also marks which parts are
verified from code and which are inferred.

**2. There is no diarization metric anywhere in the loop.** The whole point of
this checkpoint is speaker attribution without a diarization stage — and
neither this repo's STT bench nor the HF Open ASR Leaderboard scores
diarization at all. No DER, no cpWER; the quality rows are WER/UER over
reference text. Microsoft publishes no DER either — their evaluation ships as
an image in the model card, not as numbers. So the interesting property of
this model is unvalidated by anything in this PR, and would need a
speaker-attributed corpus and a cpWER/DER scorer that the bench does not have.

## The streaming route

### How it is exposed

Baseten's WebSocket transport puts a truss's socket behind one **fixed
client-facing route, `/websocket`**, and proxies that route to the single
container path `docker_server.predict_endpoint` names. Both halves matter, and
the second is what makes this package work: *the gateway route is fixed, the
container path is the truss author's choice.*

Verified from code rather than assumed:

- `_ws_path` in
  [`modalities/stt/spec.py`](../../../.github/actions/run-b10-bench/modalities/stt/spec.py)
  is what the bench uses to pick a connect path: a `bench.ws_path`
  declaration wins; otherwise `runtime.transport.kind == websocket` resolves
  to `/websocket`; otherwise the realtime protocols' own default
  (`/sync/v1/realtime`, which is for plain-HTTP vLLM trusses). This package
  declares no `ws_path`, so it resolves to `/websocket` on the strength of the
  `transport` block in `config.yaml`.
- [`stt/voxtral-mini-4b/latency`](../../voxtral-mini-4b/latency) is the live
  precedent for this exact shape — `docker_server` + vLLM +
  `is_websocket_endpoint: true` + `transport.kind: websocket` +
  `predict_endpoint: /v1/realtime` — and it benches green on
  `voxtral_realtime`. That the container path is free is the same fact twice
  over: `whisper-large-v3-streaming` uses `/websocket` for it,
  `nemotron-asr-streaming-0.6b` uses `/v1/realtime`.

So `predict_endpoint` moves from `/v1/chat/completions` to `/v1/realtime`,
and `/v1/realtime` is a route this package adds.

### Why a translator was needed

Upstream's socket and the bench's dialect are unrelated protocols. Read at the
pinned plugin commit (`vllm_plugin/asr_streaming_server.py`, the `/v1/stream`
handler) and, for the bench side, from this repo's own server for that
contract —
[`stt/qwen3-asr-1.7b-streaming/latency/model/model.py`](../../qwen3-asr-1.7b-streaming/latency/model/model.py),
whose docstring records that "the stt-benchmark `qwen_realtime` client works
unchanged" against it:

| | upstream `/v1/stream` | b10-bench `qwen_realtime` |
|---|---|---|
| handshake | first frame **must** be JSON (`receive_json`) | optional bare JSON frame, or `transcription_session.update` |
| audio | **binary** frames, little-endian **float32**, **24 kHz** | **text** JSON, `input_audio_buffer.append` with **base64 PCM16 @ 16 kHz** |
| end of turn | text frame, literally `end` | `{"type": "input_audio_buffer.commit"}` |
| per chunk | `{"chunk": k, "text", "segments"}` | `{"type": "transcription", "is_final": false, "segments": [...]}` |
| end of turn reply | `{"done": true, "total_chunks", "text", "segments", "srt"}` | the same `transcription` shape with `is_final: true` and `is_end_of_audio_flush` |

They differ on every axis — framing, sample format, sample rate, end-of-turn
signal, and response schema — so exposing `/v1/stream` directly would have
produced a routed socket that no bench protocol can speak. Declaring a
protocol that does not resolve is worse than not declaring one: it **skips**,
and a skipped bench is a green CI run that measured nothing.

### What `data/serve.py` does

It is a protocol shim and nothing more. It imports upstream's own module,
wraps its `create_app`, mounts `/v1/realtime`, and hands off to upstream's
`main()` — arg parsing, `ServerConfig`, logging and uvicorn all stay
upstream's, so the engine, the session and the geometry are unchanged. The
route drives upstream's own `StreamingSession` (via `_new_session`), cuts
windows with upstream's `split_windows`, and renders text with upstream's
`chunk_segments`, in the same loop `/v1/stream` uses: emit a window whenever
one is complete, advance by the chunk boundary so the lookahead overlap stays
buffered, and pad the tail to a full window at the end of the turn.

Three things it adds:

- **Dialect.** `input_audio_buffer.append` / `.commit` /
  `transcription_session.update` in, `transcription` messages out, partials
  replace-style (each message re-renders the whole turn, so a client displays
  the latest and drops the previous one). Unknown message types and unknown
  handshake keys are ignored and logged, the same forward-compatible posture
  the qwen3 truss takes.
- **Resampling.** Every b10-bench STT protocol sends 16 kHz PCM16; this
  checkpoint's windows are cut at 24 kHz. `scipy.signal.resample_poly` is
  stateless, so resampling each arriving frame alone would leave a filter
  transient at every frame boundary; the shim carries 192 input samples of
  context per side (overlap-save) and emits only the middle, cutting blocks to
  a multiple of the denominator so the input and output clocks stay locked and
  the ratio never drifts. Checked offline against a one-shot resample of the
  whole signal: identical to < 1e-5 across 7 s of noise fed in randomly-sized
  frames, and a 440 Hz tone comes out at 440 Hz.
- **Turn boundaries.** A VibeVoice session is append-only, so a *turn is a
  session*: `commit` finalizes and the next `append` opens a fresh upstream
  session, which also releases the finished session's windows from the
  multimodal cache. There is **no server-side VAD** on this route — unlike the
  qwen3 truss, which endpoints on silence — so a client that never commits
  never gets an `is_final` message.

Speaker attribution survives the translation. Upstream's `chunk_segments`
already splits the model's `Speaker N:` labels out of the text into a
`Speaker` field, so each emitted segment carries plain text in `text` (which
is what a WER scorer should see) and the label in an extra `speaker` field
alongside. Nothing is stripped or invented.

### What is *not* established

- **Long sessions are unprobed.** This route has now run on a GPU —
  `run-abcda730b5ac` deployed on this card, upgraded to a socket and benched
  green, after four green runs on the removed H100 arm — so the
  shim's protocol loop, window geometry and resampler are no longer
  offline-only claims (they were also exercised against a stubbed session, 21
  checks, all passing). What no run has tested is *duration*: bench clips are
  seconds long, while `--max-audio-windows 512` allows ~25 minutes, and it is a
  long session that would actually probe the multimodal-cache ceiling the
  concurrency setting is built around. TTFP, which this list previously carried
  as an inference from the geometry, is now measured — see
  [Chunk geometry](#chunk-geometry).
- **Streaming WER is not comparable to the chat route's.** Both receive 16 kHz
  audio, but the resample chains differ (this shim's polyphase upsample versus
  ffmpeg + librosa on the chat route), and upsampling cannot restore the band
  above 8 kHz that a 16 kHz source never carried. The checkpoint is a 24 kHz
  model.
- **Whether the chat route is still reachable** through the `/sync` proxy once
  the truss is served over the WebSocket transport. It is still served inside
  the container; it is no longer `predict_endpoint`, and this PR does not test
  it.

## Chunk geometry

Read from the checkpoint, never guessed — `preprocessor_config.json` carries
`chunk_frames: 22`, `lookahead_frames: 4`, `target_sample_rate: 24000`,
`speech_tok_compress_ratio: 3200`. That is a 7.5 Hz frame rate
(24000 / 3200), so each window is **2.933 s of new audio plus 0.533 s of
lookahead** the next window repeats. **Both released checkpoints — the 7B and
the 1.5B — carry the same `chunk_frames: 22` and `lookahead_frames: 4`**, read
from `preprocessor_config.json` on each. The server still reads the geometry
rather than taking a flag, because serving a checkpoint at another's geometry
does not raise — it just transcribes worse.

**Measured TTFP is ~3.2 s, and the gate is the chunk alone, not the chunk plus
the lookahead.** On this card (`run-abcda730b5ac`) mean TTFP is
**3207 ms at c=1**, 3231 ms at c=4 and 3382 ms at c=16 — flat enough across a
16x concurrency increase to be a floor rather than queueing. That is **below**
the 3.467 s that chunk + lookahead would require, and 3207 ms − 2933 ms leaves
**~274 ms of compute**, which is what emitting on chunk completion predicts.
So the 0.533 s lookahead is **context for the window, not a gate on the first
emit**, and this measurement replaces the earlier inferred "~3.5 s hard
floor", which was wrong. The floor is structural to the released geometry: no
tuning pass, larger GPU or concurrency change removes the wait for 2.933 s of
audio to exist — the removed H100 arm measured 3133-3142 ms at c=1 across
three repeats, about 70 ms below this card and inconsequential against a
~3.2 s structural floor.

The paper describes a lower-latency **15-frame** configuration. 15 frames at
7.5 Hz is **2.0 s of chunk**, and the paper's **~1.53 s** figure is the
*expected* (mean) first-emit latency for that geometry — 2.0 / 2 + 0.533,
averaging over where speech starts within a chunk — not a chunk length. Both
figures are **unverified from here** (arxiv.org is egress-blocked in these
sessions), and that calculation assumes a lookahead-gated emit, which the
measurement above does not support for the released geometry. Either way **the
15-frame geometry is in neither released checkpoint**, so ~3.1 s is what this
package can offer today.

## Accelerator: RTX PRO 6000, chosen over H100 on measured cost

**This package ships on `instance_type: RTX-PRO-6000` because that is the card
it was measured to be cheapest on, at no measurable cost in throughput or
quality.** An H100 arm of this exact package — same image, same plugin commit,
same weights, same `start_command`, same `predict_concurrency` — was benched
alongside it and has been removed. Both sets of numbers are tabulated on
[PR #395](https://github.com/basetenlabs/model-registry/pull/395); the
comparison, at the smoke profile's top rung of c=16, was:

| | RTX PRO 6000 (ships) | H100 (benched, removed) |
|---|---|---|
| Instance | `RTX-PRO-6000`, $4.0002/hr | `H100` x1, $6.4998/hr |
| Aggregate realtime at c=16 | **7.65x** | 6.64x / 6.91x / 8.09x (three repeats) |
| **$/concurrent-stream-hour** | **~$0.52** (`$4.0002 ÷ 7.65`) | **$0.80-$0.98** |
| Mean TTFP at c=1 | 3207 ms | 3133-3142 ms |
| `mu-bench-full` / `pipecat` macro WER | 17.15% / 3.16% | 16.29-20.11% / 2.27-3.16% |
| Request errors | 0.00% at every level | 0.00% at every level |

**Throughput did not measurably fall on the cheaper card.** The H100 arm's
three repeats of one identical config spread **6.64-8.09x** aggregate realtime
at c=16 — a ~22% noise floor — and this card's **7.65x** sits inside that band.
The two cards are therefore **indistinguishable on throughput from this
evidence**, in either direction; the honest claim is not that RTX PRO 6000 is
as fast but that this bench cannot tell them apart. Because capacity did not
measurably drop, the swap keeps nearly the whole **38.5%** hourly price
difference ($6.4998 → $4.0002) instead of trading it against lost throughput,
which moves this package from last in this repo's paced streaming-STT set into
the peer band of **$0.47-$0.53** that `qwen3-asr-1.7b-streaming`,
`voxtral-mini-4b` and the two `whisper-large-v3*-streaming` packages occupy —
all of them on RTX PRO 6000 too, so that is a like-for-like comparison on one
card.

**Quality and TTFP are likewise indistinguishable, or negligibly different.**
This card's WER falls inside the H100 arm's ranges on both sets, and its
sample-error rate (4.69% on both) inside the H100 spread. Mean TTFP at c=1 is
~70 ms higher, which is real but negligible against the ~3.2 s floor the
checkpoint's chunk geometry imposes; at c=16 this card is in fact *lower*
(3382 ms against 3453-3577 ms).

**One caveat travels with the comparison:** the two arms benched
**concurrently** on the run that produced these numbers — this repo's bench
concurrency group is per preset directory, so a second preset does not
serialize against the first. They ran on different accelerators and therefore
different nodes, so GPU contention is unlikely, but client-side contention is
not ruled out. Since the throughput difference is inside the H100 noise band
anyway, the defensible claim is the **price**, which follows from the published
hourly rates and a throughput figure that is statistically the same.

**vLLM 0.14.1 does serve on Blackwell (sm_120), and this is now empirical.**
The image pin is not free to move — `data/patch.py` asserts on anchor strings
inside the pinned plugin commit's `vllm_plugin/model.py`, and
[upstream's deploy doc pins `vllm/vllm-openai:v0.14.1`](https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-vllm-asr-streaming.md)
itself — so this was the blocking question before the card was tried. It is
settled by the run: the pinned image deployed, captured CUDA graphs and served
on RTX PRO 6000 with 0.00% request errors across the sweep. The build evidence
agrees —
[`CMakeLists.txt` at `v0.14.1`](https://github.com/vllm-project/vllm/blob/v0.14.1/CMakeLists.txt)
sets `CUDA_SUPPORTED_ARCHS "7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0"` and
[`docker/Dockerfile` at `v0.14.1`](https://github.com/vllm-project/vllm/blob/v0.14.1/docker/Dockerfile)
builds the published image with
`torch_cuda_arch_list='7.0 7.5 8.0 8.9 9.0 10.0 12.0'` on CUDA 12.9.1, and
**12.0 is sm_120**, so the kernels ship inside the image rather than only in
later source. Two residual notes, named rather than hidden: sm_120 has no
FlashInfer or FA3 path, so attention falls back to FlashAttention-2/Triton,
which is a different kernel than the H100 arm ran; and the known sm_120 kernel
gaps of that era are in NVFP4/MXFP4 MoE paths
([vllm#31085](https://github.com/vllm-project/vllm/issues/31085)), which this
bf16 dense checkpoint never enters. Every other vLLM truss on `RTX-PRO-6000`
in this repo runs a much newer vLLM (0.20.2-0.22.1), so the version pin had no
local precedent — the run is the precedent now.

**What is still not fixed by the cheaper card:** the **~3.2 s TTFP floor**,
which is structural to the checkpoint's chunk geometry and which no
accelerator moves, and **diarization, which is unmeasured anywhere in this
loop** — the property that makes this model worth having is validated on
neither card. See
[Read this before trusting a green bench](#read-this-before-trusting-a-green-bench).

## Hardware

- **GPU:** RTX PRO 6000 (single), named as `resources.instance_type:
  RTX-PRO-6000` rather than requested through `accelerator` + cpu/memory —
  which is what the other RTX-PRO-6000 trusses here do, and which pins the
  host side instead of leaving it to a scheduler. 96 GiB VRAM; 17.35 GB of
  bf16 weights at `--gpu-memory-utilization 0.85` leaves ~60 GB for KV cache
  and activations.
- **System memory:** 116 GiB, from the SKU. `cpu`, `memory` and `accelerator`
  are ignored when `instance_type` is set, and 116 GiB comfortably covers the
  64 GiB this package needs — `--mm-processor-cache-gb 16` is a host-side
  ceiling stacked on top of weight loading and concurrent ffmpeg decodes, and
  the sibling's 32 GiB would not be enough.
- **Session cap:** `--max-model-len 32768` with `--max-audio-windows 512`.
  A window is ~26 audio tokens, so 512 windows is ~13.3k tokens of audio plus
  text — about **25 minutes** of continuous session (512 × 2.933 s).

## Why this is not `vllm serve`

The sibling [`stt/vibevoice-asr/latency`](../../vibevoice-asr/latency) runs
`vllm serve` and proxies chat completions. A streaming checkpoint cannot:
it transcribes one chunk off a *growing* interleaved sequence, so a request is
a **session**, and stock `vllm serve` has nowhere to keep one — a client there
would have to re-send every audio window on every chunk. Upstream's answer is
`vllm_plugin.asr_streaming_server`, a FastAPI app that holds the same vLLM
`AsyncLLM` in-process, keeps each session's windows engine-side behind stable
uuids so the prefix cache stays warm, and adds the WebSocket.

`start_command` boots `data/serve.py`, which imports that module and adds one
route (see [What `data/serve.py` does](#what-dataservepy-does)); upstream's
`main()` still owns arg parsing, `ServerConfig` and uvicorn. Either way this
skips upstream's `start_streaming_server.py`, whose first steps apt-install
and pip-install at boot — work that belongs in `system_packages` and
`requirements`. The two steps of that launcher which do matter are handled:

- **Streaming-checkpoint check** — the server calls
  `ChunkGeometry.from_pretrained` itself and fails if the checkpoint has no
  `chunk_frames`.
- **Tokenizer generation is skipped on purpose.** This checkpoint already
  ships `<|text_chunk_end|>` at id 151665 in `added_tokens.json`, which is
  exactly the condition upstream's launcher checks before deciding to
  regenerate. Regenerating would rewrite files in the read-only weights mount
  for no gain.

## Scaling: no `--dp`, route whole sessions

Upstream refuses `--dp` outright, and the reason is structural: a session's
audio windows live in **one replica's prefix cache**, so round-robin would
send its later chunks to a replica that never saw the earlier ones. To use N
GPUs, run **one server per GPU and route whole sessions** with session
affinity; use `--tensor-parallel-size` only to split one model across GPUs.
This package is single-replica-per-server, TP 1.

### `predict_concurrency` on the socket

`predict_concurrency` is **16**, and the streaming route is why it moved from
the 4 the chat route was tuned to. On a WebSocket truss a "request" is a whole
**session** — one connection held open for the duration of the audio — so this
number now caps concurrent *sessions*, not whole-clip requests. The sweep that
produced 4 (realtime factor peaking at c=4 across {1, 4, 16}, b10-bench
`run-ddaef0464259`, and the tables below) measured chat-route request queueing
through one uvicorn worker; that reading does not carry over to a route where
each unit of concurrency is a live socket. Keeping 4 would have capped the
socket at four live sessions while the STT smoke profile sweeps
c ∈ {1, 4, 16}, so its c=16 cell would have measured the cap rather than the
model — and the request-error rate is the one gate that can fail a smoke
bench.

16 is the smoke profile's top rung, deliberately no higher. Upstream is
explicit that the multimodal cache (`--mm-processor-cache-gb`, 16 GB here)
"must hold every in-flight session's audio windows" and should be raised
*before* concurrency is, and on this route a session's windows stay resident
for the session's whole life — an eviction there is upstream's "Cache miss"
**error**, not a slowdown. Bench clips are seconds long, so a smoke run will
not probe that ceiling at all; the safe concurrency for many long
simultaneous sessions is unmeasured, and the lever to raise first is the cache,
not this number.

**16 is now measured on the streaming route; 4 is not.** `run-abcda730b5ac`
swept c ∈ {1, 4, 16} over the socket on this card and every cell completed 16
sessions with 0.00% request errors and no "Cache miss" failures, so the cap did
not bind the top cell and the multimodal cache held. What that does **not**
show is that 16 live streams are served well: at c=16 each session ran at only
**0.48x** the pace the bench fed it, against 0.93x at c=1 and 0.84x at c=4, so
16 oversubscribes one replica — aggregate capacity is about **7-8 concurrent
real-time streams** (7.65x aggregate realtime), which is also where the removed
H100 arm landed (6.64-8.09x across three repeats). 16 is therefore a defensible
setting rather than an optimal one, and because the sweep samples only
c ∈ {1, 4, 16} the knee between 4 and 16 is unlocated and the right value is
still unmeasured.

## API

### Streaming (WebSocket) — the routed endpoint

Connect to the Baseten WebSocket route for your environment or deployment;
Baseten proxies it to the container's `/v1/realtime`:

```
wss://model-{MODEL_ID}.api.baseten.co/environments/production/websocket
```

Authenticate with `Authorization: Api-Key $BASETEN_API_KEY`. The wire format is
the same OpenAI-realtime-style dialect the
[qwen3-asr-1.7b-streaming](../../qwen3-asr-1.7b-streaming/latency) truss
speaks, so a client written for that one works here.

```python
import asyncio, base64, json, os, wave
import websockets

URL = "wss://model-<id>.api.baseten.co/environments/production/websocket"
KEY = os.environ["BASETEN_API_KEY"]
SR, CHUNK = 16_000, int(16_000 * 0.1) * 2  # 100 ms of PCM16

async def transcribe(path: str):
    with wave.open(path, "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2
        pcm = w.readframes(w.getnframes())

    async with websockets.connect(URL, additional_headers={"Authorization": f"Api-Key {KEY}"}) as ws:
        await ws.send(json.dumps({"streaming_params": {"enable_partial_transcripts": True}}))

        async def send():
            for i in range(0, len(pcm), CHUNK):
                await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                          "audio": base64.b64encode(pcm[i:i + CHUNK]).decode()}))
                await asyncio.sleep(0.1)                       # 1x real time
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        sender = asyncio.create_task(send())

        try:
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=120))
                if msg.get("type") != "transcription":
                    continue
                for seg in msg["segments"]:
                    print(f"[{'final ' if msg['is_final'] else 'partial'}] "
                          f"{seg['start_time']:.1f}-{seg['end_time']:.1f} "
                          f"Speaker {seg['speaker']}: {seg['text']}")
                if msg.get("is_end_of_audio_flush"):
                    break
        finally:
            sender.cancel()

asyncio.run(transcribe("example.wav"))
```

Send PCM16 little-endian mono. 16 kHz is the default and is upsampled
server-side to the checkpoint's 24 kHz; a client that can capture at 24 kHz
should say so in the handshake (`streaming_params.input_sample_rate: 24000`)
and skip the resample entirely. Frame size is free — unlike streaming Whisper,
there is no VAD frame to align to.

Handshake blocks, all optional, sent as a bare first frame or nested under a
`transcription_session.update`'s `session`:

| block | key | effect |
|---|---|---|
| `streaming_params` | `enable_partial_transcripts` | per-chunk partials (default `true`) |
| `streaming_params` | `input_sample_rate` | input rate in Hz (default `16000`) |
| `whisper_params` | `prompt`, `keyterms` | mapped to VibeVoice's hotword/context slot |
| `vibevoice_params` | `context_info`, `max_tokens`, `temperature`, `top_p`, `repetition_penalty` | passed to upstream's `TranscribeRequest` |

`audio_language`, `show_word_timestamps` and `streaming_vad_config` are
accepted and **inert**: this checkpoint forces no language, ships no aligner,
and this route runs no VAD. Anything else is ignored with a server-side
warning.

Each `transcription` message carries the whole turn so far —
`segments[].{start_time, end_time, text, speaker}` plus `is_final`,
`transcription_num` and `language_code` (always `null`; the checkpoint reports
no language). Partials are replace-style: display the newest message and
discard the one before it. Segment times are arithmetic on the chunk
boundary, as upstream computes them.

### Chat completions — served, but no longer the routed endpoint

`/v1/chat/completions` is still served inside the container, and this is the
shape of a request to it, but `predict_endpoint` is now the socket and
**whether the `/sync` proxy still reaches this route under the WebSocket
transport is untested here** (see
[What is *not* established](#what-is-not-established)):

```python
from openai import OpenAI

client = OpenAI(
    api_key="<BASETEN_API_KEY>",
    base_url="https://model-<MODEL_ID>.api.baseten.co/environments/production/sync/v1",
)

response = client.chat.completions.create(
    model="vibevoice",
    messages=[{"role": "user", "content": [
        {"type": "audio_url", "audio_url": {"url": "https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav"}},
        {"type": "text", "text": "Transcribe this audio."},
    ]}],
    max_tokens=256,
    temperature=0,
)
print(response.choices[0].message.content)
```

The assistant message content is a JSON-stringified array of
speaker-attributed segments; `stream=True` returns them as SSE deltas as each
chunk finalizes. Audio can also be passed inline as
`data:audio/wav;base64,<...>`.

**Hotwords** are supported — the server pulls a hotword block out of the text
part of the message. Arbitrary instruction text is ignored rather than passed
through, so a plain prompt cannot derail the transcription.

## Known limitations

- **The chat route is no longer `predict_endpoint`.** `/predict` is the
  WebSocket now. `/v1/chat/completions`, upstream's own `/v1/stream`,
  `/v1/transcribe`, `/v1/transcribe_batch` and `/v1/config` are all still
  served inside the container, but pass-through routes only
  `predict_endpoint`, so `/v1/realtime` is the one reachable socket and the
  chat route's reachability through `/sync` is untested here.
- **No server-side VAD on the socket.** Turns are delimited by an explicit
  `input_audio_buffer.commit`; a client that streams forever without
  committing gets partials and never a final. The qwen3 streaming truss
  endpoints on silence instead — do not assume that behaviour here.
- **`example_model_input` is a chat-route body, and `/predict` is now the
  socket.** It is kept because `bench`'s `api_model_name` resolves the serving
  alias `vibevoice` from it, and because the config validator wants a
  playground example — but the playground will send this JSON to a WebSocket
  route. The streaming trusses in this repo (`qwen3-asr-1.7b-streaming`,
  `whisper-large-v3-streaming`, `voxtral-mini-4b`) ship no
  `example_model_input` at all; dropping it here is a reviewer's call, not a
  silent one.
- **End of chunk is `<|text_chunk_end|>`, not EOS.** The server sets
  `stop_token_ids=[151665]` per chunk. If the tokenizer files ever disagreed
  on that id, every chunk would run to `max_tokens` and come back empty with a
  200 — so the server verifies the id at startup and raises
  `tokenizer layout mismatch` instead. It fails loud, not silently. This
  checkpoint's `added_tokens.json` has it at 151665, so the check passes.
- **`max_tokens` is clamped server-side** to 1–2048, default 256, per chunk.
  The sibling needed a client-side `max_tokens ≈ 20 × audio_sec + 200` rule
  because VibeVoice does not reliably emit EOS and an over-generous budget
  produces a repetition loop; here the per-chunk stop token plus the clamp
  bound it for you.
- **Container formats are *not* restricted to WAV/FLAC** — this is a genuine
  divergence from the sibling, whose m4a-returns-400 limitation comes from
  vLLM 0.14.1's audio loader. The streaming server never uses that loader: it
  decodes with ffmpeg, which sniffs the container from the bytes, then
  resamples with librosa (which is what the streaming checkpoints were
  evaluated with, and differs from ffmpeg's resampler by ~1% RMS). mp3 and
  webm/opus decode fine. Untested here, but the code path is unambiguous.
- **One audio clip per request** on the chat route — the first `audio_url`
  part wins; a request with none returns 400.
- **`audio_url` must be http/https** and the body is size-capped by the
  server; other schemes return 400.
- **Cold start** is dominated by weight load; weights are pre-baked into the
  image so there is no HF download at boot.

## About `data/patch.py` and `data/serve.py`

Two files ship in `data/`, for different reasons. `serve.py` is ours to keep:
it is the protocol shim described above, and it exists because b10-bench and
upstream disagree about the wire, not because upstream is broken. `patch.py`
is the opposite — a defect workaround that should disappear.


### `patch.py`

The plugin needs **one** in-place fix on vLLM 0.14.1: KV-cache delegation
forwarders (`get_kv_cache_spec` and a `.model` property) on
`VibeVoiceForCausalLM`, which is a wrapper around `self.language_model`.
Without them vLLM's KV-cache discovery finds zero attention layers and startup
dies with `IndexError` on `available_gpu_memory[0]`. Verified still needed
against the pinned plugin commit: neither name exists in its `model.py`.

The sibling's other two patches are deliberately **not** carried over. Both are
forward-compat shims that cannot fire on a pinned 0.14.1 — `get_data_parser`
for vLLM 0.21+, and the `mm_data` → `mm_data_items` rename for 0.15+ — and the
second is worse than useless here: the sibling passes `--skip-mm-profiling` so
its rewritten body is dead code, whereas the streaming server's
`AsyncEngineArgs` does not set `skip_mm_profiling`, which would make it live
code on the dummy-profiling path with no upstream-behaviour guarantee.

The script is idempotent and `assert`s on its anchor string, so a plugin change
that breaks the anchor is a loud startup failure rather than a silent
miscompile. The plugin is commit-pinned, so that should only happen after a
deliberate bump. When Microsoft fixes this upstream, delete the file and drop
the `python3 /app/data/patch.py` step from `start_command`.

## Deployment

```bash
truss push --remote <your-baseten-remote> --publish
```

A Hugging Face token as the Baseten secret `hf_access_token` is declared for
consistency with the rest of the registry; the checkpoint itself is public and
ungated.