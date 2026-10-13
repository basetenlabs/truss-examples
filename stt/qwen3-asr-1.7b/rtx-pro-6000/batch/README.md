# Qwen3 ASR 1.7B

This example shows how to call a Baseten deployment using the OpenAI Python SDK to run **Qwen/Qwen3-ASR-1.7B** on an audio file.

The Truss loads weights from `BASETEN_MODEL_PATH`, which defaults to the BDN mount at
`/app/checkpoint/model`. Baseten Training deployments can override that variable with the
materialized path of a compatible full checkpoint.

## Keep HF and non-HF presets in sync

`Qwen/Qwen3-ASR-1.7B-hf` and `Qwen/Qwen3-ASR-1.7B` are the same underlying
trained model in different checkpoint formats. The HF variant is a
[conversion for native Hugging Face Transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_asr/convert_qwen3_asr_to_hf.py).

Maintain this preset together with its [HF batch counterpart](../../../qwen3-asr-1.7b-hf/rtx-pro-6000/batch/README.md).
Apply shared serving-configuration changes to both variants in the same PR,
including runtime/image versions, dependencies, GPU settings, environment
variables, startup flags, and performance patches. Preserve each variant's model repository ID,
pinned checkpoint revision, and format-specific loading requirements.

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

## Hardware and cold start

- **GPU:** RTX PRO 6000 (single)
- **Cold start:** about 2 minutes.

`data/start.sh` applies `data/patch.py`, in-place edits to the installed vLLM described in
that file, before starting `vllm serve`.