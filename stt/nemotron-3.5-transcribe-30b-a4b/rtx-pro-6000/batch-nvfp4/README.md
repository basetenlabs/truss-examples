# Nemotron-3.5-Transcribe-30B-A4B — Batch, NVFP4 (RTX PRO 6000)

Offline/batch transcription with NVIDIA's NVFP4 export of Nemotron 3.5 Transcribe
(`nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4@66c6adc`, 21.6 GB, ungated) on one
RTX PRO 6000, served with vLLM 0.32.0 and NeMo Speech's `nemo_speechlm` plugin. This is
the model's default batch listing. The BF16 preset is [`../batch`](../batch). How this
configuration was chosen is recorded in the Model Scout study archive,
[`model-scout/experiments/stt/nemotron-3.5-transcribe-30b-a4b/rtx-spec-decode-study/`](../../../../model-scout/experiments/stt/nemotron-3.5-transcribe-30b-a4b/rtx-spec-decode-study/)
(record `rtxspec-e33-nvfp4-mtp8-enc`).

## Configuration

| Setting | Value |
| --- | --- |
| Checkpoint | `BASETEN_MODEL_PATH=/app/checkpoint/nvfp4`, `NEMOTRON_KV_CACHE_DTYPE=fp8`, `VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel` |
| Speculative decoding | MTP, 8 draft tokens (fixed in start.sh) |
| Serving | `NEMOTRON_API_SERVER_COUNT=4`, `NEMOTRON_GPU_MEMORY_UTILIZATION=0.88`, `NEMOTRON_MAX_NUM_BATCHED_TOKENS=32768`, `NEMOTRON_MAX_NUM_SEQS=384`; prefix caching off |
| Long audio | runaway guard on; chunking above 2,400 s into 600 s chunks; `NEMOTRON_CHUNK_DECODE_WORKERS=8` |
| Audio encoder | `NEMOTRON_ENCODER_CUDA_GRAPHS=1`, `NEMOTRON_ENCODER_COMPILE=1` (with the Triton FlexAttention kernel), `VLLM_ENGINE_READY_TIMEOUT_S=1800` |
| Proxy retry | `NEMOTRON_PROXY_RETRY=0` (opt-in; see below) |
| Logging | `NEMOTRON_LOG_LEVEL` unset (INFO). Per-request diagnostics log at DEBUG |

## Performance

1-hr podcast audio, server p90, one RTX PRO 6000:

| 1-hr | RTFx | p90 |
| --- | --- | --- |
| c16 | 1,752 | 32.5 s |
| c32 | 2,096 | 53.8 s |

WER (naive): mu-bench macro 22.45%, pipecat 1.63%, TED-LIUM long-form 5.3-5.4%.

## Fine-tuned checkpoints

Training replaces the checkpoint mount and repoints `BASETEN_MODEL_PATH`; start.sh serves
whatever directory it names. A BF16 training checkpoint loads with these defaults; set
`NEMOTRON_KV_CACHE_DTYPE=auto` to give it the BF16 KV cache (fp8 KV with a BF16 checkpoint
was not measured).

## Long audio (server-side chunking)

`data/chunk_proxy.py` and `data/long_audio.py` are the H100 batch preset's chunker
(#1195 at 054fc0e); here the runaway guard also runs with chunking off. While the proxy
runs (chunking or the guard on), it listens on :8000 and vLLM on 127.0.0.1:8001. With
`NEMOTRON_CHUNK_MAX_SECONDS=0` and the guard off, start.sh execs vLLM on :8000 directly.

```
audio > TRIGGER s ─ decode once (worker process, 16 kHz mono)
                  ─ cut at the quietest 200 ms in the SEARCH s before each MAX s limit
                  ─ next chunk starts OVERLAP s before the cut
                  ─ all chunks to vLLM at once (file:// WAVs)
                  ─ no tiny last chunk: a tail under 20% of MAX is folded into the
                    previous chunk (if that fits TRIGGER) or the chunks are evened out
                  ─ stitch: semi-global word alignment of the overlap (case/punct
                    ignored, slips allowed), joined mid-overlap so each word is kept
                    once; per-seam and suspicious-chunk log lines
                  ─ one chat.completion; finish_reason "length" if any chunk hit it;
                    header X-Audio-Chunks: N
audio <= TRIGGER s ─ one engine request (URL audio is fetched by the proxy and
                     passed as file://; inline audio is forwarded unchanged)
```

| Variable | Default | Effect |
| --- | --- | --- |
| `NEMOTRON_CHUNK_MAX_SECONDS` | `600` (`0` = off) | longest chunk, at least 30 s |
| `NEMOTRON_CHUNK_TRIGGER_SECONDS` | `2400` | only longer audio is chunked; must be >= MAX |
| `NEMOTRON_CHUNK_OVERLAP_SECONDS` | `5` | overlap before each cut |
| `NEMOTRON_CHUNK_SEARCH_SECONDS` | `30` | how far back the cut may move to find a pause |
| `NEMOTRON_MAX_AUDIO_SECONDS` / `_MB` | `10800` / `1536` | per-request limits at the proxy |
| `NEMOTRON_CHUNK_DECODE_WORKERS` | `8` | processes decoding long files |

`max_tokens` applies per chunk. CTC word timestamps are moved onto the file's clock
by time ownership (chunk i owns up to its cut), independently of the text stitch;
speaker labels are per chunk.

Tests (synthetic audio, fake engine; not run by CI):
`uv run --with pytest --with pytest-asyncio --with numpy --with aiohttp --with av==19.0.2 pytest stt/nemotron-3.5-transcribe-30b-a4b/rtx-pro-6000/batch-nvfp4/tests`.

## Runaway guard (on by default)

`data/runaway.py` from #1195 (054fc0e). `NEMOTRON_RUNAWAY_GUARD=1` puts the proxy
in front of vLLM even with chunking off, so every single-audio request goes through
the proxy's file path (unchunked streaming requests then get one delta at the end).

- `max_tokens` per engine request = min(client's, ceil(audio_s x 8) + 256).
- A unit (file or chunk) is anomalous on finish_reason `length` under that cap,
  over 8 words/s, over 2.5x the request's median words/s (3+ chunks), or a loop.
- It is retried once as two halves cut at a pause (chunks >= 60 s, unchunked
  files >= 600 s); still anomalous gets the model card's `cleanup_generation_loops`
  and finish_reason `length`. Header `X-Transcript-Guard` reports what happened.
- Knobs: `NEMOTRON_RUNAWAY_TOKENS_PER_S` (8), `_MARGIN_TOKENS` (256),
  `_SPLIT_MIN_SECONDS` (600), `_CHUNK_SPLIT_MIN_SECONDS` (60).

## API server processes

`NEMOTRON_API_SERVER_COUNT` (default `4`; `1` passes no flag) sets vLLM's `--api-server-count`.
In vLLM 0.32 every frontend process builds its own app from the same arguments,
so `--middleware` (CTC timestamps) and `VLLM_PLUGINS=nemo_speechlm` load in each;
the CTC middleware reaches the engine through `collective_rpc`, whose replies vLLM
routes back to the calling process, and collects alignments by per-call job id.
With more than one process vLLM turns off its periodic stats log lines (including the
speculative-decoding line) and the IPC multimodal cache (already off:
`--mm-processor-cache-gb 0`). The Prometheus counters on `/metrics` remain, including
`vllm:spec_decode_*` for draft acceptance.

## Audio-encoder CUDA graphs and compile (on by default)

| Variable | Default | Effect |
| --- | --- | --- |
| `NEMOTRON_ENCODER_CUDA_GRAPHS` | `1` | NeMo Speech's opt-in `EncoderGraphRunner` (`speechlm2/vllm/salm/encoder_graphs.py`) via `--hf-overrides` `encoder_cuda_graphs`: the audio encoder replays from CUDA graphs captured at the first encoder call. A failed capture logs `Encoder CUDA graph capture failed` and runs eagerly |
| `NEMOTRON_ENCODER_COMPILE` | `1` | Needs graphs. Also `encoder_compile`: torch.compile of every encoder block (bf16-rounding-level numerics change) |

With compile on, start.sh writes a `.pth` file into the venv that loads `data/encoder_flex.py`
into the vLLM processes. It makes the encoder's FlexAttention calls pass
`kernel_options={"BACKEND": "TRITON"}`, the main flex_attention kernel. Without it, inductor
picks its flex_decoding kernel for clips under 128 encoder tokens, which broke short-clip
transcripts. The vLLM argv is the same either way.

Compile adds about 75 s to a cold engine init, which can pass vLLM's 600 s
`VLLM_ENGINE_READY_TIMEOUT_S`, so this preset sets it to `1800`.

## Proxy retry (opt-in)

`NEMOTRON_PROXY_RETRY=1` (default `0`; needs the proxy) makes the chunk proxy retry the
audio-URL fetch and each engine POST once on a dropped connection, answer a failed fetch
with 502 instead of 400, and keep pooled connections idle at most 4 s (vLLM's uvicorn closes
them after 5 s). It fixes the `ServerDisconnected` HTTP 400s on audio-URL fetches. It is off
by default; turn it on per deployment.

## Logging

The proxy logs at INFO by default: only its startup config line. Per-request diagnostics
(`request_error <where> <path> status=<n> body_bytes=<n>`, connection retries, chunk
summaries and seams, runaway-guard units and retries) log at DEBUG; set
`NEMOTRON_LOG_LEVEL=DEBUG` to see them. None of them log request or transcript content (ZDR).

## Bench cells

`workload.json` keeps the batch preset's cell configs, reordered and all serial so
one 1-hour and one 30-minute request at c1 (`gate=*`) run before short-clip load:
gates, then pipecat c1/16/64/96, podcast-5min c1/16/64, podcast-30min c16,
podcast-1hr c4 and the preset's quality cell (mu-bench-full + pipecat).