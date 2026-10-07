# Orukeet

[Orukeet](https://huggingface.co/oruk/orukeet) is a 627M-parameter multilingual speech recognizer
from Oruk AI, finetuned from [NVIDIA Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3).
It replaces half of the encoder's temporal depthwise filters with 12,288 fitted, frozen Gabor
kernels and retrains the rest on multilingual and multi-accent data. Everything else is Parakeet
v3: the 24-layer FastConformer encoder, the token-and-duration transducer (TDT) decoder and the
tokenizer. It transcribes the same 25 European languages with automatic language detection,
punctuation and capitalization, and word and segment timestamps. The authors report lower WER than
Parakeet v3 on 61 of 74 tested splits; those are their measurements, not ours.

It is served here through [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) with the batched Truss
runtime from the Parakeet TDT 0.6B v3 package. The Gabor kernels are ordinary convolution weights
in the checkpoint, so no special runtime is needed.

**License:** the weights are **CC BY-SA 4.0** and keep NVIDIA's attribution for the foundation
model; the model's code is **MIT**. Commercial use is permitted. Share-alike applies to modified
weights you redistribute: a further finetune of this checkpoint must carry the same license.

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

The response is `{"transcript": "...", "text": "..."}`, with both fields holding the same text.
With `"timestamps": true` it also includes `{"timestamps": {"word": [...], "segment": [...]}}`,
where each entry carries `start`/`end` offsets in seconds, for the whole recording at any length.

### Audio length

| Duration | How it runs |
| --- | --- |
| Up to 24 min | Full attention, batched with other short clips |
| 24 min to 3 hr | Local attention with a 256-frame (20.5 s) window on each side, one clip at a time |
| Over 3 hr | Rejected with HTTP 413 before any GPU work |

Inputs larger than 1 GiB are also rejected with HTTP 413. The limits are set by
`FULL_ATTENTION_MAX_SECONDS` and `MAX_AUDIO_SECONDS` (see [Configuration](#configuration)).

**Short-only deployments.** Set `MAX_AUDIO_SECONDS` to the value of `FULL_ATTENTION_MAX_SECONDS`
(`"1440"`). Clips over 24 min then get a 413, and the replica skips loading the local-attention
model, which saves about 2.3 GiB of GPU memory.

## Sample Output

```txt
Well, I don't wish to see it any more, observed Phoebe, turning away her eyes. It is certainly very like the old portrait.
```

## Hardware

The default instance is `RTX-PRO-6000` (96 GiB). A 24 min full-attention pass peaks at about
51 GiB, so the Parakeet v3 package's `T4x4x16` (14.74 GiB) cannot serve this package's default
limits. The RTX PRO 6000 is a Blackwell (sm_120) GPU, so this package pins torch 2.8.0 (CUDA 12.8)
rather than the v3 package's torch 2.6.0 (CUDA 12.4, which has no sm_120 kernels).

## Configuration

Set these under `environment_variables` in `config.yaml`. The defaults below are what this
package ships with.

| Variable | Default | What it does |
| --- | --- | --- |
| `FULL_ATTENTION_MAX_SECONDS` | `1440` | Longest clip served with full attention (24 min) |
| `MAX_AUDIO_SECONDS` | `10800` | Longest clip accepted (3 hr); longer clips get a 413. Set it equal to `FULL_ATTENTION_MAX_SECONDS` for a short-only deployment |
| `MAX_AUDIO_INPUT_BYTES` | `1073741824` | Largest input accepted (1 GiB); larger inputs get a 413 |
| `LOCAL_ATTENTION_CONTEXT_SIZE` | `256,256` | Local-attention window, in encoder frames, left and right |
| `LONG_AUDIO_STARTUP_CHECK` | `true` | At startup, run one 24 min and one 3 hr pass on silence, and fail the deploy if either does not fit on the GPU |
| `SHORT_PATH_CUDNN`, `LONG_PATH_CUDNN` | `false` | Turn cuDNN back on for the full-attention or local-attention model |
| `MAX_BATCH_SIZE` | `16` | Most clips in one full-attention batch |
| `BATCH_BUCKET_SECONDS` | `2,4,8` | Duration buckets used to group clips into batches |
| `BATCH_WINDOW_MS` | `5` | How long the batcher waits to fill a batch |
| `PREDICT_TIMEOUT_SECONDS` | `300` | Per-request timeout |
| `NEMO_CHECKPOINT` | mounted `.nemo` | Path of the checkpoint to load |

The batching and warmup values are the ones tuned for the T4 in the Parakeet v3 package. They
have not been retuned for the RTX PRO 6000. `predict_concurrency` is set to 128 in `config.yaml`.

**Diagnostic flags are off by default, and `config.yaml` sets none of them.**
`TIMING_LOG_BATCHES` and `TIMING_LOG_REQUESTS` log timings for the first N batches or requests.
`STAGE_PROBE` times each model stage with cuDNN on and off at startup and compares transcripts on
public clips it downloads. **Never enable `STAGE_PROBE` in production:** it runs extra GPU passes
after the short-clip warmup, and the build that ran it served slower in PR-CI.

## How it works

- **Two copies of the checkpoint.** `load()` restores `orukeet-v0.1.0.nemo` twice. The first copy
  keeps full attention and serves clips up to 24 min. The second is switched once to local
  attention (`rel_pos_local_attn`, `att_context_size=[256, 256]`, conv-subsampling auto-chunking)
  and serves longer clips. Neither copy changes mode after load, so no request can run under the
  wrong attention. These settings follow the long-form recipe on the
  [Parakeet TDT 0.6B v3 model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) and NeMo's
  [Inference on long audio](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/results.html#inference-on-long-audio)
  guide; the Orukeet card does not document long-form use.
- **Memory-capped batching.** Full attention is quadratic in length, so each full-attention batch
  is capped at `(FULL_ATTENTION_MAX_SECONDS / bucket)^2` clips. Buckets up to 256 s keep
  `MAX_BATCH_SIZE`.
- **cuDNN is off on both models.** On this GPU with torch 2.8 and cuDNN 9.10, cuDNN spent
  0.6-0.7 s in the encoder the first time it saw each new input length, and real audio rarely
  repeats a length. PyTorch's own kernels are as fast once warm and gave identical transcripts on
  36 test clips.
- **Load order is fixed.** The startup capacity checks run before the short-clip warmup. With them
  after it, the next CUDA call failed with an illegal memory access. The local-attention model's
  decoder runs without CUDA graphs: when it captured its own graph, the short model's graph faulted
  the same way. The ordering is commented in `load()`.
- **Failure isolation.** A failed batch is retried one request at a time, so one bad input fails
  only its own request. If the CUDA context stops working, `is_healthy()` returns False, and the
  platform stops traffic after 30 s and restarts the replica after 60 s.
- **Concurrency.** Up to `predict_concurrency: 128` requests download and decode in parallel,
  then wait for one GPU worker thread that runs every GPU call, because NeMo's `transcribe()` is
  not thread-safe. Each waiting request holds its decoded 16 kHz waveform in host RAM, about
  230 MB per hour of audio.
- **Decoding.** ffmpeg reads each input from a temp file rather than stdin, so M4A/MP4 files that
  store their index at the end decode correctly.
- **Checkpoint.** `orukeet-v0.1.0.nemo` is mounted from Hugging Face revision
  `555136b50265a132d4cea0d35560c26fc4f657ab`, the commit the model card pins for the NeMo file.
  Nothing downloads at request time. The repo's GGUF exports are left out of the mount.
- **Metrics.** Prometheus metrics on Truss's `/metrics` endpoint keep the v3 package's
  `parakeet_request_latency_seconds{phase=...}`, `parakeet_queue_depth` and `parakeet_batch_size`
  names, labeled with `model_id` and `model_version_id`.

## Benchmark

On the RTX PRO 6000 with the default config, the PR-CI smoke run on 16 Pipecat clips gave p50
latency of 86 ms at concurrency 1 and 265 ms at concurrency 16, with no errors, against 117 ms at
concurrency 1 on the v3 package's T4 config. The batch benchmark completed every clip, including
48 one-hour recordings, with 0% errors, and scored 2.05% WER on 100 Pipecat samples. Accuracy on
clips over 24 min (local attention) has not been measured. Full results are on
[#530](https://github.com/basetenlabs/model-registry/pull/530).

Bench: `baseten_predict` → `perf.stt_transcription` + `quality.stt_transcription`.