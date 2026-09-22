"""Stream a WAV file to the Nemotron 3 diarized-transcription endpoint at real-time pace.

    export BASETEN_API_KEY=...  MODEL_ID=...
    pip install websockets numpy
    python client.py meeting.wav [--max-speakers 8] [--no-words]

Prints each closed, speaker-tagged turn once as it lands, and the live partial of every speaker
currently talking. Any WAV (mono/stereo, any rate) is converted to 16 kHz mono PCM16 locally.
"""

import argparse
import asyncio
import base64
import json
import os
import time
import wave

import numpy as np
import websockets

FRAME_MS = 100
RATE = 16_000
FRAME_BYTES = RATE * FRAME_MS // 1000 * 2


def load_pcm16(path: str) -> bytes:
    with wave.open(path, "rb") as w:
        n_ch, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise SystemExit("expected 16-bit PCM WAV")
    x = np.frombuffer(raw, dtype=np.int16).reshape(-1, n_ch).mean(axis=1)
    if rate != RATE:
        x = np.interp(np.arange(0, len(x), rate / RATE), np.arange(len(x)), x)
    return x.astype(np.int16).tobytes()


async def transcribe(pcm16: bytes, max_speakers: int, words: bool) -> dict:
    url = f"wss://model-{os.environ['MODEL_ID']}.api.baseten.co/environments/production/websocket"
    headers = {"Authorization": f"Api-Key {os.environ['BASETEN_API_KEY']}"}
    async with websockets.connect(url, additional_headers=headers, max_size=None) as ws:
        await ws.send(json.dumps({"session_id": f"demo-{int(time.time())}", "max_speakers": max_speakers, "words": int(words)}))

        async def send_audio():
            for i in range(0, len(pcm16), FRAME_BYTES):
                frame = base64.b64encode(pcm16[i : i + FRAME_BYTES]).decode()
                await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": frame}))
                await asyncio.sleep(FRAME_MS / 1000)
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        sender = asyncio.create_task(send_audio())
        printed = 0
        last = {}
        async for msg in ws:
            frame = json.loads(msg)
            if frame.get("type") == "error":
                raise SystemExit(f"server error: {frame['error']}")
            last = frame
            for seg in frame["segments"][printed:]:  # closed turns are stable: print each once
                flag = " (overlap)" if seg.get("overlap") else ""
                print(f"[{seg['speaker']} {seg['start']:6.2f}-{seg['end']:6.2f}]{flag} {seg['text']}")
            printed = len(frame["segments"])
            for p in frame.get("partial", []):
                print(f"    … {p['speaker']}: {p['text']}")
            if frame.get("is_final"):
                break
        await sender
        return last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--max-speakers", type=int, default=8)
    ap.add_argument("--no-words", action="store_true", help="omit per-word timings (smaller frames)")
    args = ap.parse_args()
    final = asyncio.run(transcribe(load_pcm16(args.wav), args.max_speakers, not args.no_words))
    print(f"\nfinal: {final['num_speakers']} speaker(s), {len(final['segments'])} turns, "
          f"{final['processed_s']:.1f} s processed")


if __name__ == "__main__":
    main()
