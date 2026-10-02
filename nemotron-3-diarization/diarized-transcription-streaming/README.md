# Nemotron 3 Diarized Transcription — streaming (WebSocket)

Live speaker-attributed transcription: NVIDIA Nemotron 3 Diarization decides who is speaking on
every 80 ms frame, and NVIDIA's multitalker Parakeet 0.6B ASR (English) transcribes each active
speaker from the same mixed audio, conditioned on that speaker's activity. The output is
speaker-tagged words with timings, per-speaker live partials, and overlap flags. Speaker labels are
final the moment they appear, and a committed word is never revised. Words show a median 0.7 s
after they are spoken. One RTX PRO 6000 holds 190 concurrent hour-long streams.

Deploy from the [Model Library](https://app.baseten.co/deploy/baseten/nemotron-3-diarized-transcription-streaming); see the
[parent README](../README.md).

## Endpoint

```
wss://model-{MODEL_ID}.api.baseten.co/environments/production/websocket
Authorization: Api-Key {BASETEN_API_KEY}
```

All messages are JSON **text** frames; audio is base64-encoded inside JSON.

## Client → server

```json
{"session_id": "call-42", "max_speakers": 8}                                       // optional handshake, first frame
{"type": "input_audio_buffer.append", "audio": "<base64 PCM16 16 kHz mono>"}      // repeat, 100 ms frames
{"type": "input_audio_buffer.commit"}                                              // end of audio: flush, final frame, close
```

| Frame | Field | Type | Default | Description |
|---|---|---|---|---|
| handshake (optional) | `session_id` | string | random | Echoed in every server frame; use it to correlate logs. |
| | `max_speakers` | integer 1–8 | 8 | Caps the per-speaker ASR instances for this connection. Fewer speakers than expected is cheaper; more than the cap are merged into existing labels. |
| | `words` | 0 or 1 | 1 | Include per-word timings in `segments`. Set 0 to cut frame size by about 3×. |
| | `overlap` | 0 or 1 | 1 | Let turns of different speakers overlap in time (see below). Set 0 for a single running tail where simultaneous speech alternates word by word. |
| | `turn_segments` | 0 or 1 | 1 | Cut `segments` into turns (below). Set 0 for NeMo's native output: one running block per speaker, `partial` and `words` absent. |
| | `partials` | 0 or 1 | 1 | Send the `partial` array. |
| append | `type`, `audio` | | | `input_audio_buffer.append`; base64 of raw little-endian 16-bit mono PCM at 16 kHz. 100 ms frames (3,200 bytes) at real-time pace. |
| commit | `type` | | | `input_audio_buffer.commit`. Flushes, sends the final frame, closes. |

## Server → client

Real frame from streaming a single-speaker Harvard-sentences WAV (`words` trimmed to three per
segment for space; 31 frames in total for 34.7 s of audio):

```json
{
  "type": "transcription",
  "is_final": false,
  "session_id": "demo",
  "processed_s": 12.32,
  "num_speakers": 1,
  "segments": [
    {
      "speaker": "speaker_0",
      "start": 1.12,
      "end": 8.08,
      "text": "The birch canoe slid on the smooth planks, glue the sheet to the dark blue background.",
      "overlap": false,
      "overlaps_with": [],
      "words": [
        {
          "w": "The",
          "start": 1.12,
          "end": 1.2
        },
        {
          "w": "birch",
          "start": 1.28,
          "end": 1.68
        },
        {
          "w": "canoe",
          "start": 1.84,
          "end": 2.24
        }
      ]
    },
    {
      "speaker": "speaker_0",
      "start": 8.24,
      "end": 11.28,
      "text": "It is easy to tell the depth over well.",
      "overlap": false,
      "overlaps_with": [],
      "words": [
        {
          "w": "It",
          "start": 8.24,
          "end": 8.32
        },
        {
          "w": "is",
          "start": 8.4,
          "end": 8.48
        },
        {
          "w": "easy",
          "start": 8.48,
          "end": 8.64
        }
      ]
    }
  ],
  "partial": [
    {
      "speaker": "speaker_0",
      "text": "These days a chicken leg",
      "start": 11.36,
      "end": 12.24
    }
  ]
}
```

The final frame carries every closed turn and an empty `partial`:

```json
{"type": "transcription", "is_final": true, "session_id": "demo", "processed_s": 34.72, "num_speakers": 1}
```

| Field | Type | Description |
|---|---|---|
| `type` | string | `transcription`, or `error` (below). |
| `is_final` | boolean | `true` only on the last frame, after `commit`. |
| `session_id` | string | From the handshake, or generated. |
| `processed_s` | number | Seconds of audio the model has stepped through. |
| `num_speakers` | integer | Distinct speakers seen so far. |
| `segments` | array | **All closed turns so far**, ordered by start (replace your view). A closed turn is immutable apart from two things: a trailing word piece may be glued on if the word straddled a chunk boundary, and `overlap` may flip to `true` when a later-closing parallel turn touches it. |
| `segments[].speaker` | string | `speaker_0`…`speaker_7`, ordered by first arrival; session-local, not an identity. |
| `segments[].start`, `.end` | number | Seconds; word boundaries are on the ASR's 80 ms frame grid. |
| `segments[].text` | string | Punctuated, cased text of the turn. |
| `segments[].overlap` | boolean | This turn overlaps another speaker's turn in time. |
| `segments[].overlaps_with` | array of strings | Which speakers. Present when `overlap` is on. |
| `segments[].words` | array | `{"w", "start", "end"}` per word. Present when `words` is on. |
| `partial` | array | One entry per speaker **currently talking**: their open tail (`speaker`, `text`, `start`, `end`), re-sent every chunk (~1.1 s) until it closes and moves into `segments`. Empty on the final frame. |

### How turns are cut

Each speaker has its own open tail. It closes into a `segments` entry when that speaker pauses for
more than 1.2 s, at sentence-final punctuation, or when other speakers have talked for ≥ 1 s or
≥ 3 words since their last word (a hand-over). A one- or two-word backchannel never splits the
running turn; it becomes its own short overlapping segment. Because tails are per speaker, two
people talking at once produce two parallel turns rather than an alternation of fragments.

A frame is sent on every ASR chunk (about every 1.12 s). Frames are replace-style, so a client
that falls behind can skip frames and lose nothing.

## Errors

One `{"type": "error", "error": "…"}` frame, then the socket closes:

| `error` | Cause |
|---|---|
| `max_speakers must be in [1, 8]` / `max_speakers must be an integer` | Bad handshake value. |
| `frame must be a JSON object` / base64 decode error text | Malformed client frame. |
| `internal error: <ExceptionName>` | Server fault, logged with a traceback. |

## Limits

- 8 speakers per connection; English only for this pairing (a multilingual pairing is in progress).
- Algorithmic latency 1.12 s (the ASR chunk); speaker activity is known ~1.04 s after the audio.
- A connection that sends no frames for 120 s is finalized and closed by the server.
- `predict_concurrency` on the deployment caps admitted connections per replica at the measured hold
  (190 hour-long streams); excess connections wait at the gateway rather than degrading live ones.

## Clients

- [`client.py`](client.py) — `python client.py meeting.wav [--max-speakers N] [--no-words]`. Resamples any WAV to 16 kHz mono, streams at real-time pace, prints closed turns as they land and each speaker's live partial.

Both read `BASETEN_API_KEY` and `MODEL_ID` from the environment.
