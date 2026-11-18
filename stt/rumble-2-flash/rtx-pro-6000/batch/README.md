# Rumble 2 Flash

Research draft for **Reson8/Rumble-2-Flash** (also published as Reson8/Serval-2-Flash), a
smaller Qwen3-ASR checkpoint in the native Hugging Face Transformers format (about 509M
bf16 parameters), released under Apache-2.0. The preset is listed in the Model Library.

The serving stack (image, `data/` patches, flags and environment) is copied unchanged from
the [Qwen3 ASR 0.6B batch preset](../../../qwen3-asr-0.6b/rtx-pro-6000/batch/README.md).
Only the weights source and the served model name differ.

The Truss loads weights from `BASETEN_MODEL_PATH`, which defaults to the BDN mount at
`/app/checkpoint/model`. The upstream repository is public (ungated), so no Hugging Face
secret is needed.

## Control tokens

The model's chat template adds control tokens to the system turn (`<digits>`,
`<nofiller>`, `<verbatim>` by default, and an optional forced language).
`/v1/chat/completions` applies that template. `/v1/audio/transcriptions` builds its own
Qwen3-ASR prompt and skips the template, so `data/patch.py` adds the tokens there: the
`B10_ASR_CONTROL_TOKENS` environment variable in `config.yaml` (default
`<digits><nofiller><verbatim>`) is prepended to the system turn of every transcription
request. A caller `prompt` that already contains `<digits>` or `<nodigits>` is sent
unchanged, as the template does. Remove the variable to send upstream's prompt with no
control tokens. The patch also strips a leaked `language <Name>` header from the output
when the `<asr_text>` tag did not survive decoding.

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
        model="Reson8/Rumble-2-Flash",
        file=f,
    )

print(transcription.text)
```

Optional fields: `language` (a language code, e.g. `en`) forces the transcription
language. `prompt` becomes the system turn. Audio longer than 30 s is transcribed in 30 s
chunks. The same deployment also serves `/v1/chat/completions` with an `audio_url`
content part, where `chat_template_kwargs` (`language`, `digits`, `filler`, `verbatim`)
select the control tokens.

## Hardware

- **GPU:** RTX PRO 6000 (single)

`data/start.sh` applies `data/patch.py`, in-place edits to the installed vLLM described in
that file, before starting `vllm serve`.