# Nemotron 3 Diarization — batch (HTTP)

Diarize a recording in one request: a file URL or base64 audio in, speaker turns out. The latency
profile is chosen per request. Independent requests arriving within ~100 ms are batched into one
GPU call server-side, so throughput scales with concurrency: one RTX PRO 6000 sustains 200
six-minute files per minute.

Deploy from the [Model Library](https://app.baseten.co/deploy/baseten/nemotron-3-diarization-batch); see the [parent README](../README.md) for
model details, accuracy and the other presets.

## Endpoint

```
POST https://model-{MODEL_ID}.api.baseten.co/environments/production/predict
Authorization: Api-Key {BASETEN_API_KEY}
Content-Type: application/json
```

## Request

```json
{
  "diarization_input": {
    "audio": {"url": "https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav"},
    "latency": "offline"
  }
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `diarization_input` | object | yes | Wrapper object. A request without it is rejected with HTTP 400. |
| `diarization_input.audio` | object | yes | Exactly one of `url` or `audio_b64`. |
| `diarization_input.audio.url` | string | one of | Public or presigned URL. Fetched server-side with a 120 s timeout. |
| `diarization_input.audio.audio_b64` | string | one of | Base64 of the audio file bytes (the whole file, not raw PCM). Validated as base64. |
| `diarization_input.latency` | string | no | `offline` (default, 30.4 s buffer, best accuracy), `low` (1.04 s) or `ultralow` (0.32 s). |

Audio may be any codec `ffmpeg` can decode from a stream (WAV, FLAC, MP3, OGG, WebM/Opus, …), any
sample rate, mono or stereo; it is decoded and resampled to 16 kHz mono on the server. See Errors
for the MP4/M4A caveat. Use FLAC or a
compressed format for `audio_b64` to stay under Baseten's request-size limit on long files.

## Response

Real response for the 11-second `jfk.wav` sample above, `offline` profile (server time 936 ms, of
which 527 ms was fetching the URL):

```json
{
  "turns": [
    {
      "start": 0.29,
      "end": 2.31,
      "speaker": "speaker_0"
    },
    {
      "start": 3.26,
      "end": 4.56,
      "speaker": "speaker_0"
    },
    {
      "start": 5.37,
      "end": 10.63,
      "speaker": "speaker_0"
    }
  ],
  "segments": [
    "0.290 2.310 speaker_0",
    "3.260 4.560 speaker_0",
    "5.370 10.630 speaker_0"
  ],
  "num_speakers": 1,
  "latency": "offline",
  "batch_n": 1,
  "timing": {
    "fwd_ms": 7,
    "h2d_ms": 0,
    "mel_ms": 1,
    "steps_ms": 5,
    "graph_bs": 4,
    "nonfinite_preds": 0,
    "nonfinite_state": 0,
    "pred_max": 1.0,
    "pred_min": 0.0,
    "core_dtype": "bfloat16",
    "batch_ms": 7,
    "batch_n": 1,
    "batch_audio_s": 11.0,
    "gap_ms": 1477,
    "queue_ms": 401,
    "decode_ms": 125,
    "audio_s": 11.0,
    "post_ms": 1,
    "fetch_ms": 527,
    "total_ms": 936
  }
}
```

| Field | Type | Description |
|---|---|---|
| `turns` | array | Speaker turns, ordered by start time. `start`/`end` in seconds from the beginning of the file; `speaker` is `speaker_0` … `speaker_7`, numbered in order of first appearance. Turns of different speakers may overlap. Zero-length turns are dropped. |
| `turns[].start`, `turns[].end` | number | Seconds, 10 ms resolution. |
| `turns[].speaker` | string | Session-local label, not an identity across files. |
| `segments` | array of strings | The same turns in NeMo's `"start end speaker"` text form. |
| `num_speakers` | integer | Number of distinct speakers found (1–8). |
| `latency` | string | The profile that ran. |
| `batch_n` | integer | How many concurrent requests shared this GPU call (1 when alone). |
| `timing` | object | Server-side profile of this request, milliseconds. The useful ones: `fetch_ms` (URL download or base64 decode), `decode_ms` (ffmpeg), `queue_ms` (wait for the batch window), `fwd_ms` (model), `post_ms` (turn extraction), `total_ms`; `audio_s` is the decoded duration. The remaining keys (`h2d_ms`, `mel_ms`, `steps_ms`, `graph_bs`, `gap_ms`, `batch_audio_s`, `core_dtype`, `pred_*`, `nonfinite_*`) are engine diagnostics and may change between releases. |

## Errors

Client errors return HTTP 400 with a JSON body `{"error": "<message>"}`:

| `error` | Cause |
|---|---|
| `request must be {'diarization_input': {'audio': {...}}}` | Missing wrapper or audio object. |
| `audio requires 'url' or 'audio_b64'` | Neither given. |
| `audio_b64 is not valid base64: …` | Bad encoding. |
| `could not fetch audio.url: HTTP Error 406: Not Acceptable` (or 403/404/timeout) | The host refused the server's plain `GET` (some hosts gate on `User-Agent`), the object does not exist, or the download exceeded 120 s. Use a presigned object-store URL, or send the file as `audio_b64`. |
| `audio could not be decoded: …` | `ffmpeg` rejected the bytes. Audio is decoded from a pipe, so **MP4/M4A/MOV containers whose index (`moov`) sits at the end of the file cannot be read** — remux them (`ffmpeg -movflags +faststart`) or send WAV/FLAC/MP3/OGG. |
| `latency must be one of ['offline', 'low', 'ultralow']` | Unknown profile. |

Server faults return HTTP 500 and are logged with a traceback.

## Limits

- Up to 8 speakers per file; more speakers are merged into the nearest existing label.
- A request waits at most 30 minutes for a GPU slot and 10 minutes for post-processing before
  failing; in practice a one-hour file returns in a few seconds.
- The server batches at most 32 files per GPU call and waits up to 1.5 s to fill a batch, so the
  first request in a quiet period pays up to ~100 ms of batch-window latency.

## Clients

- [`client.py`](client.py) — `python client.py --url https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav` or `python client.py --file local.flac` (base64), with `--latency`.
- [`curl.sh`](curl.sh) — `./curl.sh https://…/audio.wav [offline|low|ultralow]`.

Both read `BASETEN_API_KEY` and `MODEL_ID` from the environment.
