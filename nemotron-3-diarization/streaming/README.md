# Nemotron 3 Diarization — streaming (WebSocket)

Live speaker diarization over a WebSocket. Stream 16 kHz mono PCM16 audio in 100 ms frames; the
server drives NVIDIA's streaming Sortformer loop with per-connection state and sends back the
current speaker-turn list on every processed chunk. Speaker identity is carried forward across the
connection with no re-clustering, so a label never changes once it appears. One RTX PRO 6000 holds
560 concurrent hour-long streams at the `low` profile, or 200 at `ultralow`.

Deploy from the [Model Library](https://app.baseten.co/deploy/baseten/nemotron-3-diarization-streaming); see the [parent README](../README.md).

## Endpoint

```
wss://model-{MODEL_ID}.api.baseten.co/environments/production/websocket
Authorization: Api-Key {BASETEN_API_KEY}
```

All messages are JSON **text** frames. Audio travels base64-encoded inside JSON, not as binary
frames.

## Client → server

```json
{"latency": "low"}                                                                 // optional handshake, first frame
{"type": "input_audio_buffer.append", "audio": "<base64 PCM16 16 kHz mono>"}      // repeat
{"type": "input_audio_buffer.commit"}                                              // end of audio: flush, final frame, close
```

| Frame | Field | Type | Description |
|---|---|---|---|
| handshake (optional, must be first) | `latency` | string | `low` (default, 1.04 s buffer), `ultralow` (0.32 s) or `offline` (30.4 s). Fixed for the connection. |
| | `threshold` | number | Per-speaker activity threshold in (0, 1); default 0.5. Lower finds more speech and more overlap, higher is stricter. |
| append | `type` | string | `input_audio_buffer.append` |
| | `audio` | string | Base64 of raw little-endian 16-bit mono PCM at 16 kHz. Any frame size works; 100 ms (3,200 bytes) is a good default. Send at real-time pace for a live source. |
| commit | `type` | string | `input_audio_buffer.commit`. The server processes the remaining audio, sends the final frame and closes the socket. |

The handshake may be omitted; the first `append` then starts a `low` session with the default
threshold. The handshake keys may also ride on the first `append` frame.

## Server → client

Real frames from streaming a 33.6 s single-speaker WAV at the `low` profile (47 frames in total,
one every 0.72 s of audio):

```json
{"type": "diarization", "is_final": false, "processed_s": 8.64, "num_speakers": 1, "turns": [{"start": 0.45, "end": 3.19, "speaker": "speaker_0"}, {"start": 4.16, "end": 6.44, "speaker": "speaker_0"}, {"start": 7.81, "end": 8.64, "speaker": "speaker_0"}]}
```
```json
{"type": "diarization", "is_final": true, "processed_s": 33.6, "num_speakers": 1, "turns": [{"start": 0.45, "end": 3.19, "speaker": "speaker_0"}, {"start": 4.16, "end": 6.44, "speaker": "speaker_0"}, …], "is_end_of_audio_flush": true}
```

| Field | Type | Description |
|---|---|---|
| `type` | string | `diarization`, or `error` (below). |
| `is_final` | boolean | `true` only on the last frame, sent after `commit`. |
| `is_end_of_audio_flush` | boolean | Present and `true` on the final frame: the trailing buffer was flushed. |
| `processed_s` | number | Seconds of audio the model has stepped through so far. |
| `num_speakers` | integer | Distinct speakers seen so far (1–8). |
| `turns` | array | **The complete current turn list** — replace your view with it, do not append. Each turn: `start`, `end` (seconds, 10 ms resolution), `speaker` (`speaker_0`…`speaker_7`, ordered by first arrival). Closed turns are stable; the last turn of a speaker who is still talking grows on each frame. Turns of different speakers may overlap. |

A frame is sent for every chunk the model steps: every 0.72 s of audio at `low` (9 frames of 80 ms),
0.24 s at `ultralow`, 27 s at `offline`. Frames are replace-style, so a client that falls behind can drop intermediate
frames and lose nothing.

## Errors

The server sends one `{"type": "error", "error": "…"}` frame and closes the socket for:

| `error` | Cause |
|---|---|
| `latency must be one of ['offline', 'low', 'ultralow']` | Unknown profile in the handshake (verified: `{"type": "error", "error": "latency must be one of ['offline', 'low', 'ultralow']"}` then close). |
| `threshold must be in (0, 1)` | Out-of-range threshold. |
| `frame must be a JSON object` / base64 decode error text | Malformed client frame. |
| `capacity: this replica is at its GPU budget for low sessions (…); retry on another replica` | Admission control: the replica is full. Reconnect; the platform routes to another replica when one exists. |
| `internal error: <ExceptionName>` | Server fault, logged with a traceback. |

## Limits

- 8 speakers per connection.
- One profile per connection; open a new connection to change it.
- Admission: a replica admits sessions while `low + 2.8 × ultralow + 0.2 × offline ≤ 560`; beyond
  that it returns the capacity error instead of degrading every stream.
- Idle connections (no frames) are closed by the platform gateway after its ping timeout; send
  `commit` when you are done rather than abandoning the socket.

## Clients

- [`client.py`](client.py) — `python client.py meeting.wav --latency low`. Resamples any WAV to 16 kHz mono, streams it at real-time pace, prints the turn list as it grows.

Both read `BASETEN_API_KEY` and `MODEL_ID` from the environment.
