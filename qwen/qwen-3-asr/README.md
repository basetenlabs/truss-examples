# Qwen3 ASR 1.7B

This example shows how to call a Baseten deployment using the OpenAI Python SDK to run **Qwen/Qwen3-ASR-1.7B** on an audio file.

The Truss loads weights from `BASETEN_MODEL_PATH`, which defaults to the BDN mount at
`/app/checkpoint/model`. Baseten Training deployments can override that variable with the
materialized path of a compatible full checkpoint.

## Prerequisites

- Python 3.9+
- OpenAI Python SDK installed:

```bash
pip install openai
```

## Example: Transcribe an audio file

```python
from openai import OpenAI

model_id = ""  # place model ID here

client = OpenAI(
    api_key="BASETEN-API-KEY",
    base_url=f"https://model-{model_id}.api.baseten.co/environments/production/sync/v1"
)

with open("audio.wav", "rb") as f:
    transcription = client.audio.transcriptions.create(
        model="Qwen/Qwen3-ASR-1.7B",
        file=f,
    )

print(transcription.text)
```

## Sample Output
```txt
Uh huh. Oh yeah, yeah. He wasn't even that big when I started listening to him, but and his solo music didn't do overly well, but he did very well when he started writing for other people.
```

Optional fields: `language` (a language code, e.g. `en`) forces the transcription
language; when omitted, the model detects the language itself and returns an
empty transcript for audio with no speech. `prompt` takes names or terms to bias
spelling. Audio longer than 30 s is transcribed in 30 s chunks, so files up to
about an hour work in one request. The same deployment also serves
`/v1/chat/completions` with an `audio_url` content part, the request shape of
the previous version of this preset.