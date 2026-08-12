# Voxtral Mini 3B

This example shows how to call a Baseten deployment to run
**mistralai/Voxtral-Mini-3B-2507** for batch (pre-recorded) transcription.

Voxtral Mini 3B (~4.7B parameters including the audio encoder, Apache 2.0) is
Mistral's compact speech understanding model. In its dedicated transcription
mode it handles audio up to about 30 minutes and automatically detects the
spoken language across 8 supported languages: English, French, German,
Spanish, Italian, Portuguese, Dutch, and Hindi. This is the batch/non-realtime
variant — for streaming transcription see the separate Voxtral Mini 4B
Realtime package.

The deployment exposes the OpenAI-compatible transcription endpoint:

```bash
curl -X POST \
  "https://model-<MODEL_ID>.api.baseten.co/environments/production/sync/v1/audio/transcriptions" \
  -H "Authorization: Api-Key <BASETEN_API_KEY>" \
  -F model=mistralai/Voxtral-Mini-3B-2507 \
  -F file=@audio.wav
```

## Example: Transcribe with the OpenAI Python SDK

```bash
pip install openai
```

```python
from openai import OpenAI

model_id = ""  # place model ID here

client = OpenAI(
    api_key="BASETEN-API-KEY",
    base_url=f"https://model-{model_id}.api.baseten.co/environments/production/sync/v1"
)

with open("audio.wav", "rb") as f:
    transcription = client.audio.transcriptions.create(
        model="mistralai/Voxtral-Mini-3B-2507",
        file=f,
    )

print(transcription.text)
```

## Sample Output
```txt
Mary had a little lamb, its fleece was white as snow, and everywhere that Mary went, the lamb was sure to go.
```

## Notes

- Serving stack: vLLM `docker_server` (`vllm/vllm-openai:v0.22.0-cu129`) — Voxtral is
  natively supported by vLLM's audio path, and the HF repo ships mistral-format
  artifacts only (`params.json` + `tekken.json`), so the start command uses the
  upstream-recommended `--tokenizer-mode mistral --config-format mistral
  --load-format mistral` trio.
- Hardware: `H100_40GB:1`; `predict_concurrency: 256` — ~9.5GB bf16 weights fit
  comfortably in a 40GB MIG slice alongside the 32k context; concurrency matches the
  batch vLLM STT siblings and is tuned from CI benchmark evidence.
- Bench: `openai_transcriptions` → batch STT perf/quality kinds via the
  OpenAI-compatible `/v1/audio/transcriptions` route. No `example_model_input`: the
  multipart file-upload request cannot be expressed as JSON (same as other
  transcriptions-endpoint entries).