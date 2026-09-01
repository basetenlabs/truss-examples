# Parakeet TDT 0.6B v3

[NVIDIA Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) is a 627M-parameter
FastConformer-TDT transducer for multilingual speech recognition, served here through
[NVIDIA NeMo](https://github.com/NVIDIA/NeMo). It transcribes 25 European languages with automatic
language detection, native punctuation and capitalization, and word/segment timestamps. It tops the
Artificial Analysis non-streaming ASR leaderboard for speed (~906x real-time) at competitive accuracy
(~4.5% AA-WER), and handles long audio: up to ~24 minutes in one pass with full attention, and up to
~3 hours with local attention.

Released under **CC-BY-4.0** — free for commercial use with attribution to NVIDIA.

## Example: Transcribe an audio URL

```python
import requests

model_id = ""  # place model ID here

resp = requests.post(
    f"https://model-{model_id}.api.baseten.co/environments/production/predict",
    headers={"Authorization": "Api-Key BASETEN-API-KEY"},
    json={"audio_url": "https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav"},
)

print(resp.json()["transcript"])
```

### Input options

| Field | Type | Description |
| --- | --- | --- |
| `audio_url` | string | URL of an audio file (any format ffmpeg can decode; resampled to 16 kHz mono) |
| `audio_b64` | string | Base64-encoded audio bytes, alternative to `audio_url` |
| `timestamps` | bool | Optional — also return word- and segment-level timestamps |

With `"timestamps": true` the response includes
`{"timestamps": {"word": [...], "segment": [...]}}` where each entry carries `start`/`end` offsets
in seconds.

## Sample Output

```txt
Well, I don't wish to see it any more, observed Phoebe, turning away her eyes. It is certainly very like the old portrait.
```

## Notes

- Serving stack: NVIDIA NeMo (`nemo_toolkit[asr]`) behind a custom Truss `model/model.py` —
  the FastConformer-TDT transducer architecture has no vLLM or SGLang implementation, so the
  registry's vLLM `docker_server` pattern does not apply. Weights load from the pinned `.nemo`
  checkpoint mounted from Hugging Face; no download happens at request time. Common audio formats
  decode in memory, with an ffmpeg pipe fallback for other codecs.
- Hardware: `T4x4x16` (Turing, sm_75, 16 GiB VRAM) — pinned explicitly so the scheduler does not
  resolve resource constraints to the more expensive T4x8x32 shape. T4 has fp16 tensor cores but
  no bf16 or FP8, so transcription runs under fp16 autocast while model weights remain in their
  checkpoint dtype.
- Concurrency: downloads and CPU decoding overlap at `predict_concurrency: 8`; one GPU worker owns
  NeMo's mutable decoder state. `MAX_BATCH_SIZE=16` remains available for runtime tuning, but the
  concurrency cap limits a replica to batches of at most eight by default. The 5 ms microbatcher
  groups compatible requests into duration buckets so short clips are not padded to unrelated long
  clips. Requests asking for timestamps retain the public contract and fall back to NeMo's
  `transcribe()` path.
- The default direct path bypasses NeMo's per-call temporary DataLoader and calls the same encoder and
  TDT decoder directly. Representative short-audio shapes are warmed at startup, long-lived startup
  objects are frozen out of Python GC scans, and deployment fails unless NeMo reports its device-side
  conditional decoder in `full_graph` CUDA-graph mode.
- Duolingo-shaped short-form benchmark (512 deterministic clips, 0.767 s mean, one T4): the selected
  SLA operating point reached 50.14 req/s / 38.47x realtime at concurrency 8 with 264.4 ms p95 and
  357.9 ms p99. The equivalent sustained batch-eight c8 validation reached 48.50 req/s / 37.21x,
  266.3 ms p95, 73% average GPU utilization, and zero errors. In the same synthetic harness, the
  controlled open-weight batch-one `transcribe()` reference reached 10.27 req/s / 7.88x at c8 with
  913.8 ms p95. That number is not Duolingo production latency. Higher-concurrency B=16 validation
  reached 115.81 req/s / 88.79x, but its 676.8 ms p95 is outside the customer's approximately 300 ms
  p95 SLO and is not the default operating point.
- A concurrency-sixteen Pipecat quality run produced hypotheses identical to both baseline and batch
  two (1.8987% WER). Peak observed framebuffer use was 5,090 MiB on the 15 GiB T4, leaving
  10,270 MiB free.
- The rejected `torch.compile` experiment is not included in the serving code or configuration.
  Torch 2.6 Inductor did not reliably compile the NeMo encoder across variable audio shapes, while
  the stateful TDT decoder already owns its own CUDA-graph implementation.
- CUDA MPS is not enabled. One Truss process loads one NeMo model instance into one CUDA context, and
  one background worker serializes GPU calls while forming duration-compatible batches. The runtime
  does not create additional model processes or explicitly manage separate CUDA streams.
- Prometheus observability is exposed on Truss's `/metrics` endpoint. The
  `parakeet_request_latency_seconds{phase=...}` histogram partitions the request critical path into
  `preprocessing`, `queueing`, `batching`, `inference`, `postprocessing`, and `total` phases;
  `parakeet_queue_depth` reports requests waiting for batch assignment; and
  `parakeet_batch_size` records the actual batch-size distribution. All three metrics use b10's
  `model_id` and `model_version_id` identity labels, populated from the platform-provided
  `BT_MODEL_ID` and `BT_MODEL_DEPLOYMENT_ID` environment variables. Prometheus also attaches the
  normal scrape labels (`cluster`, `namespace`, `pod`, `container`, and `job`). The remaining
  model-defined label is the bounded `phase` set; there are no request-level or audio-level labels.
- Bench: `baseten_predict` → `perf.stt_transcription` + `quality.stt_transcription`.
- License: CC-BY-4.0 (attribution required).