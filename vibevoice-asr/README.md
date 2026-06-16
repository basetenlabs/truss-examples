# VibeVoice-ASR

[Microsoft VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR) is a multimodal speech-to-text model deployed via [vLLM](https://github.com/vllm-project/vllm) with an OpenAI-compatible chat completions API. It transcribes audio into JSON-shaped segments with speaker labels and timestamps.

## Hardware

- **GPU:** H100 (single)
- **System memory:** 32 GiB
- **Cold start:** ~3 minutes (HF snapshot + tokenizer file generation + vLLM warmup)

## Deployment

```bash
truss push --remote <your-baseten-remote> --publish
```

A Hugging Face access token is required as a Baseten secret named `hf_access_token` to pull the model weights.

## API

The deployment exposes the OpenAI chat completions endpoint. Point any OpenAI-compatible client at it:

```python
from openai import OpenAI

client = OpenAI(
    api_key="<BASETEN_API_KEY>",
    base_url="https://model-<MODEL_ID>.api.baseten.co/environments/production/sync/v1",
)

response = client.chat.completions.create(
    model="vibevoice",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that transcribes audio input into text output in JSON format."},
        {"role": "user", "content": [
            {"type": "audio_url", "audio_url": {"url": "https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav"}},
            {"type": "text", "text": "Transcribe this audio."},
        ]},
    ],
    max_tokens=64,
    temperature=0,
)

print(response.choices[0].message.content)
```

The model returns a JSON-stringified array of segments in the assistant message content:

```json
[
  {"Start": 0.0, "End": 10.46, "Speaker": 0,
   "Content": "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country."}
]
```

You can also pass audio inline as a base64 data URI: `data:audio/wav;base64,<...>`.

## Known limitations

- **WAV (or FLAC) audio only.** vLLM 0.14.1's audio loader can't decode m4a from a buffer; m4a requests return a 400. ffmpeg-convert client-side first.
- **Size `max_tokens` proportional to audio length.** VibeVoice-ASR doesn't reliably emit an end-of-sequence token, so an over-generous `max_tokens` produces a repetition loop after the real transcript ends. A safe rule: `max_tokens ≈ 20 * audio_length_sec + 200`.
- **One audio clip per request.** The model is trained on single-audio chat messages; multiple `audio_url` items in one user turn are not supported.

## About the patch.py

`data/patch.py` applies three small in-place fixes to the installed `vllm_plugin/model.py` at container boot:

1. **KV-cache delegation forwarders** on `VibeVoiceForCausalLM` — required for vLLM 0.14.1. Without these, vLLM sees zero attention layers and crashes at startup with `IndexError: list index out of range` on `available_gpu_memory[0]`.
2. **`get_data_parser` on the Info class** — forward-compatibility shim for vLLM 0.21+ which calls the parser on the Info class instead of the Processor class.
3. **`mm_data_items` rename try/except** — forward-compatibility shim for vLLM 0.15+ which renamed the `ProcessorInputs.mm_data` field.

These are upstream plugin issues. When Microsoft fixes them, delete `data/patch.py` and remove the `python3 /app/data/patch.py` line from `config.yaml`'s `start_command`.

The patch script is idempotent and uses `assert`-checked anchor strings, so a plugin upstream change that breaks the anchors causes a clear failure at startup rather than silent miscompile.
