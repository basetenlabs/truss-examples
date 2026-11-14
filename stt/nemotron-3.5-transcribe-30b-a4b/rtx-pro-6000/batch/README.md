# NVIDIA Nemotron-3.5-Transcribe-30B-A4B — Batch (RTX PRO 6000, BF16)

[Nemotron-3.5-Transcribe-30B-A4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B)
is an offline Speech-Augmented Language Model: a hybrid Mamba-2/Transformer MoE
decoder with ASR and Sortformer speaker encoders and a bundled CTC timestamp
head. This preset runs batch transcription on one RTX PRO 6000.

## Configuration

| | |
|---|---|
| Checkpoint | `nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B` (BF16), pinned to a commit and mounted at `/app/checkpoint/model` |
| Runtime | `vllm/vllm-openai:v0.32.0` image; Python 3.13 venv from the model card's locked install of NeMo Speech main `a837f4e` (vLLM 0.32.0, Torch 2.13.0, FlashInfer 0.7.0.post1, `nemo_speechlm` plugin) |
| Serve flags | NVIDIA's model-card flags: MTP speculative decoding with 8 tokens, FP32 Mamba SSM state, ReplaySSM with the FlashInfer Mamba backend, request-level multimodal kwargs, GPU memory utilization 0.94, 65,536-token window, one audio clip of up to 120,000,000 samples (3600 s) per request |
| Scheduler | 384 sequences and 16,384 batched tokens (`predict_concurrency` 384), tuned with `/batch-bench` (the card uses 192 and 4,096) |

Deviations from the card:

- `VLLM_MAX_AUDIO_DECODE_BYTES=1610612736` (1.5 GiB): vLLM's 256 MiB default rejects 30-minute-plus 44.1 kHz stereo files.
- `VLLM_AUDIO_FETCH_TIMEOUT=120`: an hour-long `audio_url` can exceed the default fetch timeout.
- A `cuda-nvrtc-dev` build step: FlashInfer's MoE JIT needs `nvrtc.h`, which the image lacks.
- Thinking off by default: clients that omit `chat_template_kwargs` get a transcript, not a reasoning monologue.
- The CTC timestamp middleware: requests can opt into word timestamps and diarization.

`BASETEN_MODEL_PATH` names the checkpoint directory. A Baseten Training
deployment mounts its merged checkpoint and repoints it.

FlashInfer's prebuilt kernels (`flashinfer-jit-cache`, sm120f provider) cover
the fused-MoE and sampling modules, so a cold start no longer compiles them.
With SSU autotuning enabled, cold engine init took 400 / 418 s, against 749 s
without the wheel. Serving performance takes priority over cold-start time:
ReplaySSM (`checkpointing_ssu`) autotuning remains enabled, and the MoE backend
selection is unchanged. Skipping SSU autotuning was an experiment; its default
tactic was within ±4% of tuned on RTX PRO 6000, but serving parity was not
benchmarked on B200 or H100. No saved tuning file is shipped: it is coupled to
the GPU architecture, FlashInfer version and vLLM configuration.

The model-specific ReplaySSM module and Triton kernels still JIT-compile.
The BDN startup cache (`ORG_ENABLE_BDN_STARTUP_CACHE`) sets the JIT cache paths
and restores them on later starts of the same deployment. Weight loading is
additional to the vLLM `init engine` timer.

## Call the deployment

```python
import base64
from openai import OpenAI

client = OpenAI(
    api_key="BASETEN-API-KEY",
    base_url="https://model-<model_id>.api.baseten.co/environments/production/sync/v1",
)
audio = base64.b64encode(open("audio.wav", "rb").read()).decode()
response = client.chat.completions.create(
    model="nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Audio transcription. Language: as spoken. "
             "Wording: verbatim, including fillers and repetitions. "
             "Formatting: punctuation and capitalization. Output: transcript only."},
            {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio}"}},
        ],
    }],
    temperature=0,
)
print(response.choices[0].message.content)
```

For word timestamps and diarization, add `"skip_special_tokens": false` and
`"mm_processor_kwargs": {"capture_ctc_timestamps": true}` to a non-streaming
request with `n: 1`. The response carries a top-level `ctc_timestamps` object.

## Measured performance (2026-10-31)

`/batch-bench` on one RTX PRO 6000 with this config
([results](https://github.com/basetenlabs/model-registry/pull/1078#issuecomment-6539178953)).
Audio xRT is seconds of audio transcribed per wall-clock second. Podcast clips
are 44.1 kHz stereo m4a URLs.

| Cell | Concurrency | Audio xRT | Requests/s | Server p50 / p95 |
|---|---:|---:|---:|---|
| pipecat (short clips) | 64 | 756 | 76.7 | 799 / 1,034 ms |
| pipecat | 128 | 347 | 35.2 | 679 / 1,366 ms |
| podcast 5 min | 64 | 960 | 3.20 | 20.0 / 22.5 s |
| podcast 5 min | 128 | 1,119 | 3.73 | 35.1 / 36.3 s |
| podcast 5 min | 256 | 1,136 | 3.79 | 65.8 / 79.9 s |
| podcast 30 min | 16 | 384 | 0.21 | 75.6 / 76.3 s |
| podcast 30 min | 64 | 570 | 0.32 | 195.6 / 215.5 s |

- Raising the limits from the card's 192 sequences and 4,096 batched tokens
  lifted 5-minute throughput at concurrency 128 from 746 to 1,119 xRT.
- No request errors in the cells above. 30-minute audio at concurrency 128
  failed half its requests with either setting; keep long-audio concurrency at
  64 or below.
- Short-clip throughput peaks at concurrency 64. At 128 and above it drops
  while server latency stays flat, so requests wait before reaching the server.
- WER (naive, micro): pipecat 1.36% (250 samples), FLEURS 7.77% (160),
  mu-bench-full 23.14% (250). The card's limits measured 1.35%, 7.73% and
  22.87%.

## FlashInfer autotune reuse (on by default)

`NEMOTRON_AUTOTUNE_REUSE=1` lets a warm start skip FlashInfer's start-time tuning for the ops
in the autotune file the BDN startup cache restored. vLLM 0.32 loads that file but FlashInfer
0.7.0.post1 still re-profiles `checkpointing_ssu` every start (its cold-L2 policy ignores file
entries while tuning). `data/autotune_hook/sitecustomize.py` runs inside vLLM right before kernel warmup loads the file and
skips only when the file is the one vLLM resolves for this exact config (same vLLM config hash),
is the only one in the cache, and matches the installed FlashInfer version and this GPU's name.
Cold starts and mismatches tune as before and log `autotune_reuse:` with the reason. `0` opts
out. Same implementation as the H100 streaming preset, vendored here.

`NEMOTRON_DIAG_RESTART_ONCE=1` (diagnostic) starts vLLM twice in the container so the reuse is
measured on a warm start; see `data/start.sh`.

Tests: `python -m pytest stt/nemotron-3.5-transcribe-30b-a4b/rtx-pro-6000/batch/tests/test_autotune_reuse.py`.

## Limits

Audio costs about 12.5 prompt tokens per second and the transcript about 3-4
tokens per second of speech, so one request handles up to about 1 hour of
audio (a 1-hour podcast used 56,904 of 65,536 tokens). Longer or denser audio
returns HTTP 200 with `finish_reason: "length"` and a truncated transcript.
**Treat `finish_reason: "length"` as an error** and split longer recordings.

## License

OpenMDW-1.1 (model card).