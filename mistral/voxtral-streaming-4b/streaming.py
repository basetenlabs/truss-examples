import asyncio
import base64
import json
import os
import signal

import numpy as np
import sounddevice as sd
import websockets

SAMPLE_RATE = 16_000
CHUNK_MS = 100                       # send 100ms chunks
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)

WS_URL = "wss://model-4q9yn803.api.baseten.co/environments/production/websocket"
MODEL = "mistralai/Voxtral-Mini-4B-Realtime-2602"

WARMUP_SECONDS = 2.0                 # optional
SEND_COMMIT_EVERY_N_CHUNKS = 10      # optional: commit about once per second


def pcm16_to_b64(pcm16: np.ndarray) -> str:
    return base64.b64encode(pcm16.tobytes()).decode("utf-8")


async def send_warmup_silence(ws):
    """Send a little silence so the server/model warms up (optional)."""
    total = int(SAMPLE_RATE * WARMUP_SECONDS)
    silence = np.zeros(total, dtype=np.int16)

    for i in range(0, total, CHUNK_SAMPLES):
        chunk = silence[i : i + CHUNK_SAMPLES]
        await ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": pcm16_to_b64(chunk),
        }))
        await asyncio.sleep(CHUNK_MS / 1000)


async def microphone_producer(audio_q: asyncio.Queue):
    """
    Capture mic audio and push PCM16 chunks into an asyncio.Queue.
    sounddevice callback runs on a separate thread; we hop into asyncio thread safely.
    """
    loop = asyncio.get_running_loop()

    def callback(indata, frames, time_info, status):
        if status:
            # non-fatal stream warnings
            pass
        # indata is float32 in [-1, 1], shape (frames, channels)
        mono = indata[:, 0]
        pcm16 = (np.clip(mono, -1.0, 1.0) * 32767.0).astype(np.int16)
        loop.call_soon_threadsafe(audio_q.put_nowait, pcm16)

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=CHUNK_SAMPLES,
        callback=callback,
    )

    with stream:
        # run until cancelled
        while True:
            await asyncio.sleep(0.1)


async def send_audio(ws, audio_q: asyncio.Queue, stop_event: asyncio.Event):
    """Pull mic chunks from queue and send to websocket."""
    n = 0
    while not stop_event.is_set():
        try:
            pcm16 = await asyncio.wait_for(audio_q.get(), timeout=0.5)
        except asyncio.TimeoutError:
            continue

        await ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": pcm16_to_b64(pcm16),
        }))

        n += 1
        if n % SEND_COMMIT_EVERY_N_CHUNKS == 0:
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))


async def receive_text(ws, stop_event: asyncio.Event):
    """Print transcription deltas as they arrive."""
    async for msg in ws:
        if stop_event.is_set():
            break

        try:
            data = json.loads(msg)
        except json.JSONDecodeError:
            continue

        if data.get("type") == "transcription.delta":
            delta = data.get("delta", "")
            print(delta, end="", flush=True)

        # If your server emits other event types you care about, handle them here:
        # elif data.get("type") == "...": ...


async def main():
    stop_event = asyncio.Event()
    audio_q: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=50)

    def request_stop(*_):
        stop_event.set()

    # Ctrl+C handling
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    async with websockets.connect(WS_URL, extra_headers={"Authorization": f"Api-Key {BASETEN-API-KEY}"}) as ws:
        # Some servers send an initial "hello"/ack; we can just try to read once (non-fatal if it times out)
        try:
            _ = await asyncio.wait_for(ws.recv(), timeout=2)
        except Exception:
            pass

        print("\n🎙️  WebSocket connection established — you can speak into the mic now...\n")

        # Configure session/model
        await ws.send(json.dumps({"type": "session.update", "model": MODEL}))

        # Optional warmup
        await send_warmup_silence(ws)
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        # Start tasks
        mic_task = asyncio.create_task(microphone_producer(audio_q))
        send_task = asyncio.create_task(send_audio(ws, audio_q, stop_event))
        recv_task = asyncio.create_task(receive_text(ws, stop_event))

        # Wait for stop (Ctrl+C)
        while not stop_event.is_set():
            await asyncio.sleep(0.1)

        # Cleanup
        for t in (mic_task, send_task, recv_task):
            t.cancel()
        await ws.close()


if __name__ == "__main__":
    asyncio.run(main())
