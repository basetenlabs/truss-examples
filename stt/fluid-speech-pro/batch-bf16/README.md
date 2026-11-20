# Fluid Speech Pro (RTX PRO 6000, bf16, batch)

[Fluid Speech Pro](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro) is a 0.63B-parameter
FastConformer-TDT speech recognition model from FluidVoice for 25 European languages, with automatic
language detection, punctuation and capitalization, and word/segment timestamps. It is NVIDIA Parakeet
TDT 0.6B v3 plus one small residual block between the encoder's subsampling and its first Conformer
layer. This preset serves it for batch transcription on one RTX PRO 6000 in bfloat16 with
[NVIDIA NeMo](https://github.com/NVIDIA/NeMo).

**Attribution:** Fluid Speech Pro by FluidVoice. Built on NVIDIA Parakeet TDT 0.6B v3 and Moondream
Parakeet Ultra.

Released under **CC BY 4.0** (see the checkpoint's `LICENSE` and `NOTICE`): free for commercial use
with the attribution above.

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
Well, I don't wish to see it anymore, observed Phoebe, turning away her eyes. It is certainly very like the old portrait.
```

## Notes

- **Serving code.** `model/model.py` is the `stt/parakeet-tdt-0.6b-v3/batch` serving code (same
  Truss stack: py310, torch 2.8.0, `nemo_toolkit[asr]==3.0.0`, direct batched TDT forward, bf16 cast,
  cuDNN off, encoder CUDA graphs, optional encoder/decoder overlap, batches of up to 16, `baseten_predict`).
  Every serving knob in `config.yaml` except `PIPELINE_OVERLAP` (below) is the Parakeet preset's
  value, tuned on Parakeet TDT 0.6B v3 in Model Scout; see
  [that preset's README](../../parakeet-tdt-0.6b-v3/batch/README.md) for what each knob does and
  how it was measured. None has been retuned for this checkpoint yet.
- **Loader.** Plain `ASRModel.restore_from` refuses this `.nemo` (unexpected `encoder.fv_insert` keys).
  `model/fluid_speech.py` is the model authors' loader, vendored byte-identical from
  `FluidVoice-Labs/fluid-speech-pro@46027a3bffc5817ce8f9b29add0a062610134a90` (sha256
  `4a1873d8e84207c8ae19b57e0b4f85916309d33a18ad922ef204de6685b57e0e`, checked by
  `tests/test_fluid_speech.py`). It restores the `.nemo` non-strictly, adds the block as
  `encoder.fv_insert` (LayerNorm, Linear 1024→1024, depthwise Conv1d k=15, GELU, Linear, residual),
  runs it from a forward hook on `encoder.pre_encode`, then loads every tensor strictly. The repo's
  `model.safetensors` (Transformers format) leaves the block out and is not used: only
  `fluid-speech-pro.nemo` is mounted.
- **Block, bf16 and CUDA graphs.** The block is a submodule of the encoder, so the one-time bf16 cast
  (`PRECISION_MODE=full`) casts it with the encoder, and its hook runs inside `encoder.forward()`, so
  the encoder CUDA graphs (captured from `model.forward()`, preprocessor + encoder) record it. `load()`
  fails unless the block is attached, has exactly one `pre_encode` hook and shares the encoder's dtype,
  both after load and after the cast (logged as `Fluid Speech block ready after ...`). The startup
  transcript check then runs through a graph replay.
- **Clean mode** (`FLUID_CLEAN_MODE`, default `false`). The checkpoint also holds `clean_mode.*`
  low-rank deltas for edited, non-verbatim text (no repeated or filler words). Off, the loader drops
  them and the model writes verbatim text, which is the reference loader's default and what verbatim
  WER references expect. `true` folds them into the weights once at load.
- **Decoding** (`DECODING_STRATEGY`, default `greedy_batch`). The default keeps the checkpoint's
  batched greedy TDT decoder with its full CUDA graph. `malsd_batch` switches to NeMo's batched TDT
  beam search at load with `BEAM_SIZE` beams (default 4) and only the best hypothesis returned, as
  the card's Open ASR leaderboard numbers use. The beam decoder runs without CUDA graphs (no run has
  measured them); the encoder graphs still replay. `malsd_batch` requires
  `REQUIRE_FULL_CUDA_GRAPH=false`, because that check covers the greedy decoder only. The startup
  transcript and pipeline parity checks run through the beam decoder.
- **Padding.** Unlike the Conformer layers, the block's depthwise convolution does not mask padded
  frames, so the last ~7 encoder frames (~0.6 s) of a clip see the batch's padding. NeMo's own batched
  `transcribe()` behaves the same way. On a CPU check, padding one LibriSpeech clip by 2 s changed its
  encoder output by 0.3% (relative L2) and left the transcript unchanged.
- **Fail-fast.** After any CUDA error, only that batch fails and a health probe runs: synchronize,
  then one kernel read back (the same check as `stt/orukeet/batch`). If the probe errors, the CUDA
  context is unusable: `is_healthy()` turns false, new requests fail at once, and the process exits
  after `FATAL_EXIT_DELAY_SECONDS` (default 2) so the replica restarts. A sticky error (device-side
  assert, illegal memory access, launch failure) fails the probe; the Parakeet preset kept answering
  500s after such an assert. Other CUDA errors (such as `invalid argument`) can leave the context
  usable, and then the replica stays up. `runtime.health_checks` stops traffic after 30 s unhealthy
  and restarts after 60 s, which also covers a dead batch worker thread. Each failed batch logs the
  last 4 batches: rows, clip lengths, graph key, and serial or graph+overlap path.
- **Pipeline overlap** (`PIPELINE_OVERLAP`, default `false`). Overlap decodes batch N on a second
  CUDA stream while batch N+1's encoder graph replays. With it on, a CUDA fault hit 5 of 7 greedy
  Model Scout runs on PR #1377, each losing 38-93 clips until the replica restarted: an
  `IndexKernel.cu:113` index-out-of-bounds assert in the TDT label-looping greedy decode, or
  `invalid argument` at the pinned-staging sync. One of those runs was the stock
  `stt/parakeet-tdt-0.6b-v3/batch` preset (run 44467068402, assert on FLEURS uk_ua, then every later
  request on that replica failed), so the fault comes from the shared serving stack, not this
  checkpoint's extra block. With overlap off, two full 55-job runs (44487637672 and the final-head
  run 44493920088) had zero faults. The root cause is unconfirmed; the likeliest reading is a race
  between the overlapped encoder and decoder streams that corrupts the decode's indices. A zero
  cuBLAS workspace (run 44483085719) did not prevent it. Quality is the same either way (FLEURS-25
  mean 10.82%). Off costs some peak throughput: on ali-ue1-prod-1 the final run peaked at 294 req/s
  against 315-330 req/s with overlap on in earlier runs (-7 to -11%); on aws-uw2-prod-2, 243 against
  297 (-18%). Low-concurrency latency depends on the cluster, not on overlap: c1 server p50 was
  ~390 ms on ali-ue1-prod-1 and 22 ms on aws-uw2-prod-2 with overlap on or off.
  `PIPELINE_OVERLAP=true` (needs `ENCODER_CUDA_GRAPHS=true`) restores the overlapped path; the
  fail-fast above then limits a fault to one replica restart. Two clean runs with overlap off are
  thin evidence: the fault may still occur with it off, and fail-fast then limits it to one replica
  restart too.
- **Precision.** `INFERENCE_DTYPE` sets it: `bfloat16` (default) or `float32`, which serves the
  checkpoint's float32 weights uncast. Both keep the encoder CUDA graphs.
- Hardware: one RTX PRO 6000 (Blackwell, 96 GiB). Weights are mounted from the pinned Hugging Face
  revision; the repo is not gated. Nothing downloads at request time.
- Bench: `baseten_predict` → `perf.stt_transcription` + `quality.stt_transcription`.
- License: CC BY 4.0 (attribution required, above).