import asyncio
import os
import json
import queue
import numpy as np
import websockets
import sphn
import sounddevice as sd
from collections import deque

SAMPLE_RATE = 24000
FRAME_SIZE = 480  # 20ms frames for better responsiveness (24000 / 50)
CHANNELS = 1

# Queues for audio data with max size to prevent buildup
MAX_QUEUE_SIZE = 10  # Max ~200ms of audio buffered
mic_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
speaker_buffer = deque(maxlen=100000)  # Use deque for efficient speaker buffering
speaker_lock = asyncio.Lock()

# Statistics
stats = {"mic_overflows": 0, "speaker_underruns": 0, "queue_size": 0}


def audio_callback_in(indata, frames, time, status):
    """Callback for microphone input."""
    if status:
        if status.input_overflow:
            stats["mic_overflows"] += 1

    # indata is (frames, channels), convert to mono if needed
    mono = indata[:, 0] if indata.shape[1] > 1 else indata.flatten()

    try:
        mic_queue.put_nowait(mono.copy())
    except queue.Full:
        # Drop old frames if queue is full (prevents latency buildup)
        try:
            mic_queue.get_nowait()
            mic_queue.put_nowait(mono.copy())
        except:
            pass


def audio_callback_out(outdata, frames, time, status):
    """Callback for speaker output."""
    if status:
        if status.output_underflow:
            stats["speaker_underruns"] += 1

    # Pull from speaker buffer
    if len(speaker_buffer) >= frames:
        for i in range(frames):
            outdata[i, 0] = speaker_buffer.popleft()
    else:
        # Not enough data - output what we have + silence
        available = len(speaker_buffer)
        for i in range(available):
            outdata[i, 0] = speaker_buffer.popleft()
        outdata[available:, 0] = 0
        if available < frames // 2:  # Only log if significant underrun
            stats["speaker_underruns"] += 1

    stats["queue_size"] = len(speaker_buffer)


async def stream_conversation():
    """Real-time streaming conversation with PersonaPlex."""

    uri = "wss://model-lqzgyvow.api.baseten.co/environments/production/websocket"

    opus_writer = sphn.OpusStreamWriter(SAMPLE_RATE)
    opus_reader = sphn.OpusStreamReader(SAMPLE_RATE)

    api_key = os.environ.get("BASETEN_API_KEY")
    if not api_key:
        print("ERROR: BASETEN_API_KEY environment variable not set")
        return

    print(f"Connecting to: {uri}")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(
        f"Frame size: {FRAME_SIZE} samples ({FRAME_SIZE / SAMPLE_RATE * 1000:.1f} ms)"
    )
    print(
        f"Max queue size: {MAX_QUEUE_SIZE} frames ({MAX_QUEUE_SIZE * FRAME_SIZE / SAMPLE_RATE * 1000:.1f} ms)"
    )

    # List available audio devices
    print("\nAvailable audio devices:")
    devices = sd.query_devices()
    print(devices)

    # Try to find best input/output devices
    default_input = sd.default.device[0]
    default_output = sd.default.device[1]
    print(f"\nUsing input device: {default_input}")
    print(f"Using output device: {default_output}")

    try:
        # Start audio streams with smaller blocksize for better responsiveness
        input_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            callback=audio_callback_in,
            blocksize=FRAME_SIZE,
            dtype=np.float32,
            device=default_input,
        )
        output_stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            callback=audio_callback_out,
            blocksize=FRAME_SIZE,
            dtype=np.float32,
            device=default_output,
        )

        input_stream.start()
        output_stream.start()
        print("\n🎤 Microphone started")
        print("🔊 Speakers started")

        async with websockets.connect(
            uri,
            additional_headers={"Authorization": f"Api-Key {api_key}"},
            ping_interval=20,
            ping_timeout=10,
        ) as ws:
            print("\n📡 WebSocket connected, sending config...")

            # Send config as first message
            config = {
                "voice_prompt": "NATF0.pt",
                "text_prompt": "You are a helpful assistant.",
                "seed": -1,
            }
            await ws.send(json.dumps(config))

            # Wait for handshake
            print("Waiting for handshake...")
            msg = await ws.recv()

            if isinstance(msg, bytes) and len(msg) > 0:
                if msg[0] == 0x00:
                    print("✓ Handshake received! Ready to talk!\n")
                else:
                    print(f"⚠️  Unexpected first byte: 0x{msg[0]:02x}")

            # Stats monitoring
            last_stats_time = asyncio.get_event_loop().time()

            async def send_audio():
                """Send microphone audio to server."""
                accumulated = np.array([], dtype=np.float32)
                frame_count = 0

                while True:
                    try:
                        # Get all available audio from microphone queue
                        while not mic_queue.empty():
                            chunk = mic_queue.get_nowait()
                            accumulated = np.concatenate([accumulated, chunk])

                        # Send full frames
                        while len(accumulated) >= FRAME_SIZE:
                            frame = accumulated[:FRAME_SIZE]
                            accumulated = accumulated[FRAME_SIZE:]

                            # Encode and send
                            opus_writer.append_pcm(frame)
                            data = opus_writer.read_bytes()
                            if len(data) > 0:
                                await ws.send(bytes([0x01]) + data)
                                frame_count += 1

                        await asyncio.sleep(0.01)  # 10ms loop for responsiveness

                    except Exception as e:
                        print(f"\n❌ Send error: {e}")
                        break

            async def receive_responses():
                """Receive and play server responses."""
                nonlocal last_stats_time

                try:
                    async for message in ws:
                        if not isinstance(message, bytes) or len(message) == 0:
                            continue

                        msg_type = message[0]
                        payload = message[1:]

                        if msg_type == 0x01:  # Audio
                            opus_reader.append_bytes(payload)
                            pcm = opus_reader.read_pcm()
                            if pcm.shape[-1] > 0:
                                # Add to speaker buffer
                                # Clear buffer if it's getting too full (prevents latency)
                                if len(speaker_buffer) > SAMPLE_RATE * 2:  # >2 seconds
                                    speaker_buffer.clear()

                                speaker_buffer.extend(pcm)

                        elif msg_type == 0x02:  # Text
                            # Receive text but don't print it
                            text = payload.decode("utf-8")

                except Exception as e:
                    print(f"\n❌ Receive error: {e}")

            # Run send and receive concurrently
            await asyncio.gather(
                send_audio(),
                receive_responses(),
            )

    except KeyboardInterrupt:
        print("\n\n⏹️  Stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Stop audio streams
        if "input_stream" in locals():
            input_stream.stop()
            input_stream.close()
        if "output_stream" in locals():
            output_stream.stop()
            output_stream.close()
        print("\n🔇 Audio streams closed")


if __name__ == "__main__":
    print("=" * 60)
    print("PersonaPlex Voice Chat Client")
    print("=" * 60)
    print("\nPress Ctrl+C to stop\n")

    try:
        asyncio.run(stream_conversation())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
