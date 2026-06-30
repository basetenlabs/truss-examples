# Parakeet TDT 0.6B V3 — dynamic batching

A [Truss](https://truss.baseten.co/) for NVIDIA's
[`parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) English
ASR model, with **server-side dynamic batching** for high-throughput HTTP
transcription. It runs the NeMo model directly (fp16 on Turing/T4, bf16 on Ampere+)
and batches concurrent requests on the GPU.

This is the batching counterpart to [`../parakeet-tdt-0_6b-v2`](../parakeet-tdt-0_6b-v2)
(single-request).

## How batching works

Incoming requests are decoded to audio on the request thread, then handed to a small
pipeline:

- a **collector** thread drains the request queue (with an adaptive collect window),
  pads the batch, and runs the mel preprocessor on the GPU;
- an **inference** thread runs the encoder + TDT decoder (with CUDA graphs) on the
  previous batch while the collector prepares the next one.

The adaptive collect window only waits for stragglers when 2+ requests are already
queued, so single-arrival tail latency is unchanged while concurrent load grows
batches above 1.

## Configuration (`config.yaml` → `environment_variables`)

| var | default | meaning |
| --- | --- | --- |
| `BATCH_INFERENCE` | `true` | enable the batched pipeline (`false` = serial) |
| `MAX_BATCH_SIZE` | `64` | max requests per GPU batch |
| `BATCH_COLLECT_MS` | `20` | adaptive collect window (ms) for stragglers |
| `CUDA_GRAPHS` | `true` | CUDA graphs for the TDT decoder loop |
| `TORCH_COMPILE` | `false` | `torch.compile` the encoder |

The model weights are pulled from Hugging Face into a Baseten volume cache
(`model_cache` in `config.yaml`) for fast cold starts; `nvidia/parakeet-tdt-0.6b-v3`
is public, so no token is required. Default accelerator is `L4`.

## Deploy

```bash
truss push
```

## Call

```bash
curl -X POST https://model-<MODEL_ID>.api.baseten.co/environments/production/predict \
  -H "Authorization: Api-Key $BASETEN_API_KEY" \
  -d '{"audio_url": "https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav", "timestamps": false}'
```

Input fields:

- `audio_url` — URL fetched and decoded server-side, **or**
- `audio_base64` — base64-encoded audio bytes (wav/mp3/opus/… via ffmpeg)
- `timestamps` — `true` returns word-level timestamps (serial path; not batched)

Response: `{"transcript": "..."}`.
