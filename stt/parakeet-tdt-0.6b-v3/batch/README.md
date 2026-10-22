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
  decode in memory. Other codecs fall back to ffmpeg, which reads the upload from a temporary file so
  it can seek to an M4A index (`moov` atom) written after the audio data.
- Hardware: one RTX PRO 6000 (Blackwell, 96 GiB). Blackwell needs CUDA 12.8 kernels, so this preset
  pins torch 2.8.0 (CUDA 12.8 wheels) and NeMo 3.0.0, the same stack as `stt/orukeet/batch`.
  The encoder, decoder and joint are cast to bfloat16 once at load (`PRECISION_MODE=full`, no
  autocast) and cuDNN is off (`ENCODER_CUDNN=false`), the Model Scout round-2 settings. Concurrency,
  batch size and length bucketing were retuned on this GPU in Model Scout rounds 1 and 3; warmup
  shapes are still the values tuned on T4, and the T4 benchmark figures below were
  measured on the earlier T4 deployment (still listed as `baseten/parakeet-tdt-0-6b-v3-t4`). Startup
  logs report the installed Torch version and its compiled CUDA and cuDNN versions for deployment
  verification.
- Concurrency: the preset admits up to `predict_concurrency: 128` requests. Model Scout round 1 measured
  the earlier setting of 64; one GPU worker owns NeMo's mutable decoder state, so admitted requests
  queue for the batcher and run
  in batches of up to `MAX_BATCH_SIZE=16`. Length bucketing is on (`BATCH_LENGTH_BUCKETING=true`,
  Model Scout round 3, run 39678817538): the 5 ms microbatcher looks ahead over up to
  `BATCH_BUCKET_LOOKAHEAD` (default 4 x `MAX_BATCH_SIZE`) pending clips and builds each batch from the
  oldest clip plus the clips closest to it in length, so short clips are not padded to unrelated long
  clips. On RTX PRO 6000 this raised c=64 throughput from 45.96 to 53.79 req/s and cut server p50
  from 86 to 75 ms, with c=1/8 unchanged and identical WER. Batch-window early exit is on
  (`BATCH_WINDOW_EARLY_EXIT=true`, Model Scout round 7, run 39733446532): the 5 ms window ends as soon
  as the queue is empty and no `predict()` call is still decoding audio, instead of idling the GPU for
  the rest of it. With 60 s timed cells this raised c=64 throughput 6.0% over the mean of two
  identical baseline arms (which agreed within 0.2%), cut server p50 from 29-31 to 26 ms at c=1 and
  from 123-127 to 112 ms at c=64, and left WER unchanged (20.87% / 3.17%). The code default stays
  off. Requests asking for timestamps retain the public contract and fall back to NeMo's
  `transcribe()` path. That call rebuilds NeMo's decoding object with timestamps on, so the first
  direct-path batch after a run of timestamp batches rebuilds the load-time decoding, re-captures its
  decoder CUDA graph with one warmup batch and re-checks `full_graph` mode before decoding (logged as
  `Parakeet decoding restore finished: phase=rewarm_after_timestamps, ... elapsed_s=`).
- The default direct path bypasses NeMo's per-call temporary DataLoader and calls the same encoder and
  TDT decoder directly. Representative short-audio shapes are warmed at startup, long-lived startup
  objects are frozen out of Python GC scans, and deployment fails unless NeMo reports its device-side
  conditional decoder in `full_graph` CUDA-graph mode.
- Encoder CUDA graphs (`ENCODER_CUDA_GRAPHS`, on by default since Model Scout round 4, run 39701075887;
  ported from `stt/orukeet/batch-short`). On RTX PRO 6000, on the same cluster, they cut server p50
  from 42 to 30 ms at c=1 and from 86 to 51 ms at c=32, with throughput within noise and WER unchanged
  within noise (pipecat 3.20% both; mu-bench 20.90% vs 20.70%). The graphs
  replay the preprocessor and encoder from graphs captured at startup, one per batch size in
  `ENCODER_GRAPH_BATCH_SIZES` (up to the first covering `MAX_BATCH_SIZE`) and length step of
  `ENCODER_GRAPH_STEP_SECONDS` up to `ENCODER_GRAPH_MAX_SECONDS`. `config.yaml` captures batch sizes
  1,2,4,6,8,10,12,14,16 (270 graphs; the code default 1,2,4,8,16 gives 150). Model Scout round 9 (run
  39750926774, same cluster) adopted the even sizes: a 9-15 clip batch no longer pads to 16 rows, which
  cut the encoder median by 28-43% for 5-6 and 9-12 clip batches (batch 9: 50.7 to 32.8 ms) and server
  p50 at c=64 from 48 to 42 ms, with throughput and WER tied. Capturing 270 graphs takes 56-74 s at
  startup (150 took 34-57 s) and peak allocated GPU memory stays about 3.25 GiB. A batch is
  zero-padded to the next captured shape; anything larger or longer (for example, clips over 30 s)
  runs eagerly, so there is no input length limit. Graphs are captured in the serving dtype, before the TDT decoder's own graph; a capture
  failure is fatal at startup (the replica does not serve), and the startup transcript check runs through a graph. The encoder's
  positional table is pre-built for `ENCODER_POS_TABLE_SECONDS` (default 1440) of audio; a longer
  input rebuilds it and switches the replica to eager for good (logged as an error).
  `parakeet_encoder_batches_total{path}` counts graphed and eager batches, and `GRAPH_PARITY_BATCHES=N`
  (debug, default 0) reruns the first N graphed batches eagerly and logs transcript differences.
- Encoder/decoder overlap (`PIPELINE_OVERLAP`, on in `config.yaml` since #926, code default `false`;
  requires `ENCODER_CUDA_GRAPHS`). Model Scout round 11 (run 39767116873, same cluster, aws-uw2) lifted
  audio xRT 11.4% and round 13 (run 39777023292, same cluster, cor-ue13a) 6.4%, client-path bound,
  with WER flat. The batch worker replays batch N+1's encoder graph on one CUDA stream,
  from double-buffered pinned input, while a decode thread runs batch N's TDT decode on a second
  stream. A CUDA event hands each batch over through a queue of depth 1, and one decode runs at a
  time. Timestamps batches, decoding rebuilds, eager-encoder batches and `GRAPH_PARITY_BATCHES`
  reruns drain the pipeline and run serially. At startup one four-clip batch is decoded serially and
  twice through the pipeline, and any transcript difference fails the deploy. `TIMING_LOG_BATCHES`
  times overlapped batches with per-stream CUDA events and logs `pipeline=overlap` with each stage's
  start on a shared GPU timeline.
- Offline ceiling: the Open ASR leaderboard harness that measured this model's offline RTFx ceiling
  on this GPU is bench-only and does not ship in the serving image. See
  [`model-scout/tools/open-asr-offline-ceiling/`](../../../model-scout/tools/open-asr-offline-ceiling/).
- `NEMO_CHECKPOINT` can override the mounted `.nemo` path without changing the runtime. Startup logs
  record the selected checkpoint, installed NeMo version, restored model/encoder/decoder classes,
  and the observed versus required decoder CUDA-graph mode before and after warmup.
- Duolingo-shaped short-form benchmark on the earlier T4 deployment, with torch 2.6 and NeMo 2.7.3
  (512 deterministic clips, 0.767 s mean, one T4): the selected
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
  does not create additional model processes, and it uses separate CUDA streams only with
  `PIPELINE_OVERLAP=true`.
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