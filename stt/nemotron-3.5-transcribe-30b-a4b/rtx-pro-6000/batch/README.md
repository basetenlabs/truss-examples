# NVIDIA Nemotron-3.5-Transcribe-30B-A4B — Batch (RTX PRO 6000, BF16)

[Nemotron-3.5-Transcribe-30B-A4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B)
is an offline Speech-Augmented Language Model: a hybrid Mamba-2/Transformer MoE
decoder with ASR and Sortformer speaker encoders and a bundled CTC timestamp
head. This preset runs batch transcription on one RTX PRO 6000.

## Configuration

| | |
|---|---|
| Checkpoint | `nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B` (BF16), pinned to a commit and mounted at `/app/checkpoint/model` |
| Runtime | `vllm/vllm-openai:v0.28.0` image; Python 3.13 venv from the model card's locked install of NeMo Speech `16b0ff2` (vLLM 0.28.0, Torch 2.13.0, `nemo_speechlm` plugin) |
| Serve flags | NVIDIA's model-card flags: MTP speculative decoding with 8 tokens, FP32 Mamba SSM state, GPU memory utilization 0.94, 65,536-token window, 192 sequences, 4,096 batched tokens, one audio clip of up to 120,000,000 samples (3600 s) per request |

Deviations from the card:

- `VLLM_MAX_AUDIO_DECODE_BYTES=1610612736` (1.5 GiB): vLLM's 256 MiB default rejects 30-minute-plus 44.1 kHz stereo files.
- `VLLM_AUDIO_FETCH_TIMEOUT=120`: an hour-long `audio_url` can exceed the default fetch timeout.
- A `cuda-nvrtc-dev` build step: FlashInfer's MoE JIT needs `nvrtc.h`, which the image lacks.
- Thinking off by default: clients that omit `chat_template_kwargs` get a transcript, not a reasoning monologue.
- The CTC timestamp middleware: requests can opt into word timestamps and diarization.
- `--hf-overrides` sets the architecture to `NeMoSpeechLMForConditionalGeneration`: the public `config.json` renamed it, and the pinned NeMo Speech commit registers only the old name.

`BASETEN_MODEL_PATH` names the checkpoint directory. A Baseten Training
deployment mounts its merged checkpoint and repoints it.

Cold start is about 16-20 minutes, most of it JIT compilation and Mamba2
warmup. `start.sh` keeps the JIT caches in b10cache when it is enabled, so
later starts reuse them.

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

## Measured performance (2026-10-28)

`/batch-bench` on one RTX PRO 6000 with this config. Podcast clips are 44.1 kHz
stereo m4a URLs.

| Cell | Concurrency | Requests/s | Audio s per wall s | Server p50 / p95 |
|---|---:|---:|---:|---|
| pipecat (mean 9.6 s) | 32 | 40.2 | 386 | 761 / 1,028 ms |
| pipecat | 64 | 61.4 | 590 | 1,008 / 1,393 ms |
| pipecat | 96 | 61.7 | 591 | 1,560 / 1,859 ms |
| podcast 1 min | 2 | 1.52 | 91 | 1.32 / 1.37 s |
| podcast 5 min | 2 | 0.40 | 120 | 5.0 / 5.1 s |
| podcast 5 min | 16 | 0.97 | 290 | 16.9 / 18.6 s |
| podcast 5 min | 48 | 1.40 | 420 | 34.4 / 39.9 s |

- Long audio passes: a direct client with `max_tokens` unset got full
  transcripts (`finish_reason: stop`) for 30 minutes at concurrency 1 in
  41.7 s, and 1 hour at concurrency 4 in about 160 s.
- No errors, out-of-memory events or engine deaths in any cell.
- WER (naive, micro): mu-bench-full 19.10%, pipecat 1.30%, pipecat-5min 2.35%.

## Limits

Audio costs about 12.5 prompt tokens per second and the transcript about 3-4
tokens per second of speech, so one request handles up to about 1 hour of
audio (a 1-hour podcast used 56,904 of 65,536 tokens). Longer or denser audio
returns HTTP 200 with `finish_reason: "length"` and a truncated transcript.
**Treat `finish_reason: "length"` as an error** and split longer recordings.

## License

OpenMDW-1.1 (model card).