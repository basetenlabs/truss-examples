# Kokoro v1.0

[Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) is an open-weight TTS model with 82 million parameters that runs on a single T4 GPU. This Truss serves Kokoro v1.0 over HTTP and returns base64-encoded WAV audio at 24 kHz.

Model weights are mounted via the [Baseten Delivery Network](https://docs.baseten.co/development/model/bdn) and pinned to a specific Hugging Face commit, so cold starts skip the upstream download.

## Deploy

```bash
truss push
```

## Call the model

```json
request:
{"text": "Hello world", "voice": "af_heart", "speed": 1.0}

response:
{"base64": "<base64-encoded WAV bytes>"}
```

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `text` | string | `"Hi, I'm Kokoro."` | Text to synthesize. `KPipeline` chunks long input automatically. |
| `voice` | string | `"af_heart"` | Voice name. See [VOICES.md](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md) for the full list. |
| `speed` | float | `1.0` | Speech speed multiplier. |

The voice prefix encodes the language: `a` is American English, `b` is British English, `j` is Japanese, `z` is Mandarin, `e` is Spanish, `f` is French, `h` is Hindi, `i` is Italian, `p` is Portuguese. All voicepacks are preloaded at startup from the BDN mount, so the first request for any voice has no extra download cost.

## Language support

| Languages | Status |
| --- | --- |
| American and British English (`a`, `b`) | Works out of the box. |
| Spanish, French, Hindi, Italian, Portuguese (`e`, `f`, `h`, `i`, `p`) | Works out of the box via `espeak-ng`. |
| Japanese (`j`) | Add `misaki[ja]` to `requirements` in `config.yaml`. |
| Mandarin (`z`) | Add `misaki[zh]` to `requirements` in `config.yaml`. |

## Cold-start behavior

The first inference call after a cold start takes a few seconds while Kokoro compiles its CUDA kernels. Subsequent calls return audio in under a second.
