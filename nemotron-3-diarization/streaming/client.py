"""Stream a WAV file to the Nemotron 3 Diarization streaming endpoint at real-time pace.

    export BASETEN_API_KEY=...  MODEL_ID=...
    pip install websockets numpy
    python client.py meeting.wav --latency low

Any WAV (mono/stereo, any rate) is converted to 16 kHz mono PCM16 locally. Each server frame
carries the full current turn list; this client prints the newest turns as they change.
"""

import argparse
import asyncio
import base64
import json
import os
import wave

import numpy as np
import websockets

FRAME_MS = 100
RATE = 16_000
FRAME_BYTES = RATE * FRAME_MS // 1000 * 2  # PCM16 mono


def load_pcm16(path: str) -> bytes:
    with wave.open(path, "rb") as w:
        n_ch, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise SystemExit("expected 16-bit PCM WAV")
    x = np.frombuffer(raw, dtype=np.int16).reshape(-1, n_ch).mean(axis=1)
    if (
        rate != RATE
    ):  # linear resample; good enough for a demo, use soxr/ffmpeg in production
        t_new = np.arange(0, len(x), rate / RATE)
        x = np.interp(t_new, np.arange(len(x)), x)
    return x.astype(np.int16).tobytes()


async def stream(pcm16: bytes, latency: str) -> dict:
    url = f"wss://model-{os.environ['MODEL_ID']}.api.baseten.co/environments/production/websocket"
    headers = {"Authorization": f"Api-Key {os.environ['BASETEN_API_KEY']}"}
    async with websockets.connect(url, additional_headers=headers, max_size=None) as ws:
        await ws.send(json.dumps({"latency": latency}))

        async def send_audio():
            for i in range(0, len(pcm16), FRAME_BYTES):
                frame = base64.b64encode(pcm16[i : i + FRAME_BYTES]).decode()
                await ws.send(
                    json.dumps({"type": "input_audio_buffer.append", "audio": frame})
                )
                await asyncio.sleep(FRAME_MS / 1000)  # real-time pacing
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        sender = asyncio.create_task(send_audio())
        last = {}
        async for msg in ws:
            frame = json.loads(msg)
            if frame.get("type") == "error":
                raise SystemExit(f"server error: {frame['error']}")
            last = frame
            tail = frame["turns"][-2:]
            print(
                f"t={frame['processed_s']:6.1f}s  speakers={frame['num_speakers']}  "
                + "  ".join(
                    f"{t['speaker']}[{t['start']:.1f}-{t['end']:.1f}]" for t in tail
                )
            )
            if frame.get("is_final"):
                break
        await sender
        return last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--latency", default="low", choices=["low", "ultralow", "offline"])
    args = ap.parse_args()
    final = asyncio.run(stream(load_pcm16(args.wav), args.latency))
    print(f"\nfinal: {final['num_speakers']} speaker(s), {len(final['turns'])} turns")
    for t in final["turns"]:
        print(f"{t['start']:8.2f}  {t['end']:8.2f}  {t['speaker']}")


if __name__ == "__main__":
    main()
