# MOSS Transcribe-Diarize

[MOSS-Transcribe-Diarize 0.9B](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize)
performs multilingual transcription, speaker diarization, and timestamping in one
pass. This configuration serves it with SGLang Omni on one RTX PRO 6000.

`data/moss_pipeline.yaml` tunes SGLang Omni for that GPU: an FP8 KV cache (long-audio decode is
bound by KV reads, and the RTX PRO 6000 has about half the H100's memory bandwidth), up to 64
running requests, and larger encoder batches and request-build queues so short clips are not
starved upstream of decode. Measured against the same recipe on one H100 in bf16, with identical
b10-bench cells: 5-minute clips at 32 concurrent reach 375x real time (H100: 327) and 30-minute
clips at 16 concurrent 86x (H100: 85), with no errors.

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

## Response

`verbose_json` returns the usual OpenAI transcription envelope plus `segments`:

```json
{
  "text": "full transcript ...",
  "segments": [
    {"start": 12.34, "end": 15.67, "text": "[S1] and then we shipped it"},
    {"start": 15.90, "end": 17.02, "text": "[S0] right"}
  ]
}
```

Two things differ from what callers usually assume:

- **The speaker is an inline `[S<n>]` tag at the start of `text`**, not a field.
  Strip it yourself; there is no `speaker` key anywhere in the response. A segment
  can also arrive untagged, so treat "no tag" as a real case rather than an error.
- **Segment bounds are `start` / `end`**, not `start_time` / `end_time`. A consumer
  that assumes the longer names silently reads every segment as starting at 0.0.

Speaker labels are session-local (`S0`, `S1`, ...) and carry no identity across
requests.