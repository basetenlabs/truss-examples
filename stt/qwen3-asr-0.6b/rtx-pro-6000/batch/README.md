# Qwen3 ASR 0.6B

This example shows how to call a Baseten deployment using the OpenAI Python SDK to run **Qwen/Qwen3-ASR-0.6B** on an audio file.

The serving stack (image, `data/` patches, flags and environment) is copied from the
[Qwen3 ASR 1.7B batch preset](../../../qwen3-asr-1.7b/rtx-pro-6000/batch/README.md).

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
        model="Qwen/Qwen3-ASR-0.6B",
        file=f,
    )

print(transcription.text)
```

## Sample Output

Illustrative output format: this transcript comes from the 1.7B preset, and the 0.6B's exact wording can differ.

```txt
Uh huh. Oh yeah, yeah. He wasn't even that big when I started listening to him, but and his solo music didn't do overly well, but he did very well when he started writing for other people.
```

Optional fields: `language` (a language code, e.g. `en`) forces the transcription
language; when omitted, the model detects the language itself and returns an
empty transcript for audio with no speech. `prompt` takes names or terms to bias
spelling. Audio longer than 30 s is transcribed in 30 s chunks, so files up to
about an hour work in one request. The same deployment also serves
`/v1/chat/completions` with an `audio_url` content part.

## Hardware

- **GPU:** RTX PRO 6000 (single)

`data/start.sh` applies `data/patch.py`, in-place edits to the installed vLLM described in
that file, before starting `vllm serve`.