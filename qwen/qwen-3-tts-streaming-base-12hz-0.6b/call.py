"""WebSocket client for streaming text-input TTS with Base (voice cloning) only.

Connects to the /v1/audio/speech/stream endpoint, sends text incrementally
(simulating real-time STT output), and saves a single audio file per stream.

Voice cloning (Base task):
    Provide --ref-audio (local file) and --ref-text to clone a voice.
    Set --voice-name to cache the clone server-side so subsequent sessions
    skip the expensive embedding extraction.

Usage:
    # Voice cloning (first time: uploads + caches)
    python call.py \
        --text "Hello world. How are you?" \
        --ref-audio /path/to/reference.wav \
        --ref-text "Transcript of the reference audio." \
        --voice-name my_voice

    # Voice cloning (subsequent: uses cached voice, no ref-audio needed)
    python call.py \
        --text "Hello world. How are you?" \
        --voice-name my_voice

    # Simulate STT: send text word-by-word with delay
    python call.py \
        --text "Hello world. How are you? I am fine." \
        --voice-name my_voice \
        --simulate-stt --stt-delay 0.1

Requirements:
    pip install websockets
"""

import argparse
import asyncio
import base64
import json
import mimetypes
import os
import time
import wave

try:
    import websockets
except ImportError:
    print("Please install websockets: pip install websockets")
    raise SystemExit(1)


def _write_wav(path: str, pcm_data: bytes, sample_rate: int, channels: int) -> None:
    """Write raw PCM-16LE bytes to a WAV file."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit = 2 bytes per sample
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)


def _encode_audio_file(path: str) -> str:
    """Read a local audio file and return a base64 data URI."""
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type is None:
        ext = os.path.splitext(path)[1].lower()
        mime_map = {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".m4a": "audio/mp4",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg",
            ".aac": "audio/aac",
            ".webm": "audio/webm",
        }
        mime_type = mime_map.get(ext, "audio/wav")

    with open(path, "rb") as f:
        audio_bytes = f.read()

    b64 = base64.b64encode(audio_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


async def _delete_voice(url: str, voice_name: str) -> None:
    """Connect to the WebSocket endpoint and send a voice.delete command."""
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({"type": "voice.delete", "voice_name": voice_name}))
        raw = await ws.recv()
        msg = json.loads(raw)
        if msg.get("type") == "voice.deleted":
            print(f"Voice '{voice_name}' deleted successfully.")
        elif msg.get("type") == "error":
            print(f"Error: {msg.get('message')}")
        else:
            print(f"Unexpected response: {msg}")


async def stream_tts(
    url: str,
    text: str,
    config: dict,
    output_file: str,
    simulate_stt: bool = False,
    stt_delay: float = 0.1,
) -> None:
    """Connect to the streaming TTS endpoint and process audio responses."""
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    async with websockets.connect(url, additional_headers={"Authorization": f"Api-Key {os.getenv('BASETEN_API_KEY')}"}) as ws:
        # 1. Send session config
        config_msg = {"type": "session.config", **config}
        t_request = time.perf_counter()
        await ws.send(json.dumps(config_msg))
        print(f"Sent session config: { {k: (v[:60] + '...' if isinstance(v, str) and len(v) > 60 else v) for k, v in config.items()} }")

        # Ensure text ends with punctuation to prevent cutoff
        def ensure_ending_punctuation(t: str) -> str:
            t = t.strip()
            if t and t[-1] not in ".!?;:…。！？":
                return t + "."
            return t

        text_to_send = ensure_ending_punctuation(text)

        # 2. Send text (either all at once or word-by-word)
        async def send_text():
            if simulate_stt:
                words = text_to_send.split(" ")
                for i, word in enumerate(words):
                    chunk = word + (" " if i < len(words) - 1 else "")
                    await ws.send(
                        json.dumps(
                            {
                                "type": "input.text",
                                "text": chunk,
                            }
                        )
                    )
                    print(f"  Sent: {chunk!r}")
                    await asyncio.sleep(stt_delay)
            else:
                await ws.send(
                    json.dumps(
                        {
                            "type": "input.text",
                            "text": text_to_send,
                        }
                    )
                )
                print(f"Sent full text: {text_to_send!r}")

            # 3. Signal end of input
            await ws.send(json.dumps({"type": "input.done"}))
            print("Sent input.done")

        # Run sender and receiver concurrently
        sender_task = asyncio.create_task(send_text())

        sentence_count = 0
        ttfa: float | None = None
        sample_rate: int = 24000
        interrupted = False

        # Accumulate all PCM chunks across all sentences into a single buffer.
        all_pcm: list[bytes] = []

        try:
            while True:
                message = await ws.recv()

                if isinstance(message, bytes):
                    if ttfa is None:
                        ttfa = time.perf_counter() - t_request
                        print(f"  [TTFA] Time to first audio: {ttfa * 1000:.1f} ms")
                    all_pcm.append(message)
                else:
                    msg = json.loads(message)
                    msg_type = msg.get("type")

                    if msg_type == "voice.registered":
                        print(f"  Voice '{msg.get('voice_name')}' registered (cached={msg.get('cached')})")

                    elif msg_type == "audio.start":
                        print(f"  [sentence {msg['sentence_index']}] Generating: {msg['sentence_text']!r}")

                    elif msg_type == "audio.done":
                        idx = msg["sentence_index"]
                        sample_rate = msg.get("sample_rate", 24000)
                        print(f"  [sentence {idx}] Done")
                        sentence_count += 1

                    elif msg_type == "session.done":
                        t_total = time.perf_counter() - t_request
                        pcm_data = b"".join(all_pcm)
                        _write_wav(output_file, pcm_data, sample_rate=sample_rate, channels=1)
                        audio_duration = len(pcm_data) / (sample_rate * 2) if pcm_data else 0
                        print(f"\nSession complete: {msg['total_sentences']} sentence(s) generated")
                        print(f"  Saved {output_file} ({len(pcm_data)} PCM bytes, {audio_duration:.2f}s)")
                        if ttfa is not None:
                            print(f"  TTFA:       {ttfa * 1000:.1f} ms")
                        print(f"  Total time: {t_total * 1000:.1f} ms")
                        if audio_duration > 0:
                            print(f"  RTF:        {t_total / audio_duration:.2f}x")
                        break
                    elif msg_type == "error":
                        print(f"  ERROR: {msg['message']}")
                    else:
                        print(f"  Unknown message: {msg}")

        except (asyncio.CancelledError, KeyboardInterrupt):
            interrupted = True

        finally:
            sender_task.cancel()
            try:
                await sender_task
            except asyncio.CancelledError:
                pass

            if interrupted:
                t_total = time.perf_counter() - t_request
                print(f"\nInterrupted after {t_total * 1000:.0f} ms")
                pcm_data = b"".join(all_pcm)
                if pcm_data:
                    _write_wav(output_file, pcm_data, sample_rate=sample_rate, channels=1)
                    audio_duration = len(pcm_data) / (sample_rate * 2)
                    print(f"  Saved partial audio: {output_file} "
                          f"({len(pcm_data)} PCM bytes, {audio_duration:.2f}s)")
                else:
                    print("  No audio received yet.")
                await ws.close(code=1000, reason="Client interrupted")


def main():
    parser = argparse.ArgumentParser(description="Streaming text-input TTS client")
    parser.add_argument(
        "--url",
        default="wss://model-wgl5y7o3.api.baseten.co/deployment/q41xjd5/websocket",
        help="WebSocket endpoint URL",
    )
    parser.add_argument(
        "--text",
        required=False,
        help="Text to synthesize",
    )
    parser.add_argument(
        "--output",
        default="streaming_tts_output.wav",
        help="Output WAV file path (default: streaming_tts_output.wav)",
    )

    # Session config options (Base task only)
    parser.add_argument("--model", default=None, help="Model name")
    parser.add_argument("--language", default="Auto", help="Language")
    parser.add_argument(
        "--response-format",
        default="wav",
        choices=["wav", "pcm", "flac", "mp3", "aac", "opus"],
        help="Audio format",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed (0.25-4.0)")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Max tokens")

    # Voice cloning options (Base task)
    parser.add_argument(
        "--ref-audio",
        default=None,
        help="Path to local reference audio file for voice cloning",
    )
    parser.add_argument(
        "--ref-text",
        default=None,
        help="Transcript of reference audio (enables ICL mode). "
        "Can be inline text or a path to a .txt file.",
    )
    parser.add_argument(
        "--voice-name",
        default=None,
        help="Name for caching the voice clone server-side. "
        "On first use provide --ref-audio too; subsequent calls reuse the cache.",
    )
    parser.add_argument(
        "--x-vector-only-mode",
        action="store_true",
        default=False,
        help="Speaker embedding only mode (no ICL). Enables per-sentence "
        "streaming for cached voices, giving much lower TTFA at the cost "
        "of slightly reduced voice similarity.",
    )
    parser.add_argument(
        "--delete-voice",
        default=None,
        metavar="NAME",
        help="Delete a cached voice clone by name and exit.",
    )

    # STT simulation
    parser.add_argument(
        "--simulate-stt",
        action="store_true",
        help="Simulate STT by sending text word-by-word",
    )
    parser.add_argument(
        "--stt-delay",
        type=float,
        default=0.1,
        help="Delay between words in STT simulation (seconds)",
    )

    args = parser.parse_args()

    # Handle --delete-voice as a one-shot command
    if args.delete_voice:
        asyncio.run(_delete_voice(args.url, args.delete_voice))
        return

    # If ref-text looks like a file path, read its contents
    if args.ref_text and os.path.isfile(args.ref_text):
        with open(args.ref_text) as f:
            args.ref_text = f.read().strip()
        print(f"Read ref_text from file: {args.ref_text[:80]}{'...' if len(args.ref_text) > 80 else ''}")

    # Encode local reference audio as base64 data URI
    ref_audio_data_uri = None
    if args.ref_audio:
        if not os.path.isfile(args.ref_audio):
            print(f"Error: reference audio file not found: {args.ref_audio}")
            raise SystemExit(1)
        ref_audio_data_uri = _encode_audio_file(args.ref_audio)
        size_kb = os.path.getsize(args.ref_audio) / 1024
        print(f"Encoded reference audio: {args.ref_audio} ({size_kb:.1f} KB)")

    # Build session config (Base task only; only include non-None values)
    config: dict = {"task_type": "Base"}
    for key in [
        "model",
        "language",
        "response_format",
        "speed",
        "max_new_tokens",
        "ref_text",
        "voice_name",
    ]:
        val = getattr(args, key.replace("-", "_"), None)
        if val is not None:
            config[key] = val

    if ref_audio_data_uri is not None:
        config["ref_audio"] = ref_audio_data_uri

    if args.x_vector_only_mode:
        config["x_vector_only_mode"] = True

    try:
        asyncio.run(
            stream_tts(
                url=args.url,
                text=args.text,
                config=config,
                output_file=args.output,
                simulate_stt=args.simulate_stt,
                stt_delay=args.stt_delay,
            )
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
