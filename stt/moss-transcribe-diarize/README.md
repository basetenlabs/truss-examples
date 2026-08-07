# MOSS Transcribe-Diarize

[MOSS-Transcribe-Diarize 0.9B](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize)
performs multilingual transcription, speaker diarization, and timestamping in one
pass. This configuration serves it with SGLang Omni on one H100.

The deployment exposes the OpenAI-compatible transcription endpoint:

```bash
curl -X POST \
  "https://model-<MODEL_ID>.api.baseten.co/environments/production/sync/v1/audio/transcriptions" \
  -H "Authorization: Api-Key <BASETEN_API_KEY>" \
  -F model=OpenMOSS-Team/MOSS-Transcribe-Diarize \
  -F file=@audio.wav \
  -F response_format=verbose_json
```

Use `response_format=json` for the raw transcript or `verbose_json` for parsed
speaker segments. For long audio, pass a larger output budget such as
`-F max_new_tokens=65536`.