"""WebSocket client for streaming text-input TTS with voice cloning support.

Connects to the /v1/audio/speech/stream endpoint, sends text incrementally
(simulating real-time STT output), and saves a single audio file per stream.

Voice cloning:
    Provide --ref-audio (local file) and optionally --ref-text to clone a
    voice.  Set --voice-name to cache the clone server-side so subsequent
    sessions skip the expensive embedding extraction.

Usage:
    # Send full text at once
    python streaming_speech_client.py --text "Hello world. How are you? I am fine."

    # Simulate STT: send text word-by-word with delay
    python streaming_speech_client.py \
        --text "Hello world. How are you? I am fine." \
        --simulate-stt --stt-delay 0.1

    # VoiceDesign task
    python streaming_speech_client.py \
        --text "Today is a great day. The weather is nice." \
        --task-type VoiceDesign \
        --instructions "A cheerful young female voice"

    # Voice cloning (first time: uploads + caches)
    python streaming_speech_client.py \
        --text "Hello world. How are you?" \
        --task-type Base \
        --ref-audio /path/to/reference.wav \
        --ref-text "Transcript of the reference audio." \
        --voice-name my_voice

    # Voice cloning (subsequent: uses cached voice, no ref-audio needed)
    python streaming_speech_client.py \
        --text "Hello world. How are you?" \
        --task-type Base \
        --voice-name my_voice

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

try:
    import numpy as np
except ImportError:
    print("Please install numpy: pip install numpy")
    raise SystemExit(1)

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


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


async def benchmark_single_request(
    url: str,
    text: str,
    config: dict,
    request_id: int,
    save_path: str | None = None,
) -> dict:
    """Run a single TTS request and return timing metrics.

    If *save_path* is provided, the received PCM audio is written as a WAV file.
    """
    def ensure_ending_punctuation(t: str) -> str:
        t = t.strip()
        if t and t[-1] not in ".!?;:…。！？":
            return t + "."
        return t

    text_to_send = ensure_ending_punctuation(text)
    result = {"request_id": request_id, "ttfa": None, "total_time": None, "audio_duration": None, "error": None}
    total_pcm_bytes = 0
    all_pcm: list[bytes] = []
    sample_rate = 24000

    try:
        async with websockets.connect(
            url,
            additional_headers={"Authorization": f"Api-Key {os.getenv('BASETEN_API_KEY')}"},
        ) as ws:
            config_msg = {"type": "session.config", **config}
            t_request = time.perf_counter()
            await ws.send(json.dumps(config_msg))

            await ws.send(json.dumps({"type": "input.text", "text": text_to_send}))
            await ws.send(json.dumps({"type": "input.done"}))

            while True:
                message = await ws.recv()

                if isinstance(message, bytes):
                    if result["ttfa"] is None:
                        result["ttfa"] = (time.perf_counter() - t_request) * 1000  # ms
                    total_pcm_bytes += len(message)
                    if save_path is not None:
                        all_pcm.append(message)
                else:
                    msg = json.loads(message)
                    msg_type = msg.get("type")

                    if msg_type == "audio.done":
                        sample_rate = msg.get("sample_rate", 24000)
                    elif msg_type == "session.done":
                        result["total_time"] = (time.perf_counter() - t_request) * 1000  # ms
                        break
                    elif msg_type == "error":
                        result["error"] = msg.get("message")
                        break
    except Exception as e:
        result["error"] = str(e)

    # Calculate audio duration from total PCM bytes (16-bit mono = 2 bytes per sample)
    if total_pcm_bytes > 0:
        result["audio_duration"] = total_pcm_bytes / (sample_rate * 2)

    if save_path is not None and all_pcm:
        pcm_data = b"".join(all_pcm)
        _write_wav(save_path, pcm_data, sample_rate=sample_rate, channels=1)
        result["audio_file"] = save_path

    return result


async def run_concurrency_sweep(
    url: str,
    text: str,
    config: dict,
    concurrency_levels: list[int],
    requests_per_level: int,
    warmup_requests: int = 2,
    save_audio: bool = False,
    match_concurrency: bool = False,
) -> None:
    """Run a concurrency vs TTFA sweep benchmark."""
    audio_dir: str | None = None
    if save_audio:
        audio_dir = "benchmark_audio"
        os.makedirs(audio_dir, exist_ok=True)
        print(f"Audio outputs will be saved to {audio_dir}/")

    print(f"\n{'='*60}")
    print("Concurrency vs TTFA Sweep Benchmark")
    print(f"{'='*60}")
    print(f"URL: {url}")
    print(f"Text: {text[:50]}{'...' if len(text) > 50 else ''}")
    print(f"Concurrency levels: {concurrency_levels}")
    if match_concurrency:
        print(f"Requests per level: (matches concurrency)")
    else:
        print(f"Requests per level: {requests_per_level}")
    print(f"Warmup requests: {warmup_requests}")
    print(f"{'='*60}\n")

    # Warmup
    if warmup_requests > 0:
        print(f"Running {warmup_requests} warmup request(s)...")
        warmup_tasks = [
            benchmark_single_request(url, text, config, i)
            for i in range(warmup_requests)
        ]
        warmup_results = await asyncio.gather(*warmup_tasks)
        warmup_ttfas = [r["ttfa"] for r in warmup_results if r["ttfa"] is not None]
        if warmup_ttfas:
            print(f"  Warmup TTFA: {np.mean(warmup_ttfas):.1f}ms avg")
        print()

    results_table = []

    for concurrency in concurrency_levels:
        # Use concurrency as request count when match_concurrency is True
        num_requests = concurrency if match_concurrency else requests_per_level
        print(f"Testing concurrency={concurrency} ({num_requests} requests)...")

        # Run requests in batches of `concurrency`
        all_ttfas = []
        all_total_times = []
        all_audio_durations = []
        errors = 0

        num_batches = (num_requests + concurrency - 1) // concurrency

        for batch_idx in range(num_batches):
            batch_size = min(concurrency, num_requests - batch_idx * concurrency)
            tasks = []
            for i in range(batch_size):
                req_id = batch_idx * concurrency + i
                save_path = (
                    os.path.join(audio_dir, f"c{concurrency}_req{req_id:03d}.wav")
                    if audio_dir else None
                )
                tasks.append(
                    benchmark_single_request(url, text, config, req_id, save_path=save_path)
                )

            batch_results = await asyncio.gather(*tasks)

            for r in batch_results:
                if r["error"]:
                    errors += 1
                else:
                    if r["ttfa"] is not None:
                        all_ttfas.append(r["ttfa"])
                    if r["total_time"] is not None:
                        all_total_times.append(r["total_time"])
                    if r["audio_duration"] is not None:
                        all_audio_durations.append(r["audio_duration"])

        if all_ttfas:
            ttfa_arr = np.array(all_ttfas)
            total_avg_ms = float(np.mean(all_total_times)) if all_total_times else 0
            audio_avg_s = float(np.mean(all_audio_durations)) if all_audio_durations else 0
            # RTF = generation_time / audio_duration (lower is better, <1 means faster than real-time)
            rtf_avg = (total_avg_ms / 1000) / audio_avg_s if audio_avg_s > 0 else 0
            stats = {
                "concurrency": concurrency,
                "requests": len(all_ttfas),
                "errors": errors,
                "ttfa_avg": float(np.mean(ttfa_arr)),
                "ttfa_min": float(np.min(ttfa_arr)),
                "ttfa_max": float(np.max(ttfa_arr)),
                "ttfa_p50": float(np.percentile(ttfa_arr, 50)),
                "ttfa_p95": float(np.percentile(ttfa_arr, 95)),
                "ttfa_p99": float(np.percentile(ttfa_arr, 99)),
                "total_avg": total_avg_ms,
                "audio_avg": audio_avg_s,
                "rtf_avg": rtf_avg,
            }
            results_table.append(stats)

            print(f"  TTFA: avg={stats['ttfa_avg']:.0f}ms, p50={stats['ttfa_p50']:.0f}ms, "
                  f"p95={stats['ttfa_p95']:.0f}ms, p99={stats['ttfa_p99']:.0f}ms, RTF={rtf_avg:.2f}x")
            if errors:
                print(f"  Errors: {errors}")
        else:
            print(f"  No successful requests (errors: {errors})")

    # Print summary table
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"{'Conc':>6} | {'Reqs':>5} | {'Err':>4} | {'Avg':>7} | {'P50':>7} | {'P95':>7} | {'P99':>7} | {'Total':>7} | {'RTF':>5}")
    print(f"{'-'*6}-+-{'-'*5}-+-{'-'*4}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*5}")
    for s in results_table:
        print(f"{s['concurrency']:>6} | {s['requests']:>5} | {s['errors']:>4} | "
              f"{s['ttfa_avg']:>6.0f}ms | {s['ttfa_p50']:>6.0f}ms | {s['ttfa_p95']:>6.0f}ms | "
              f"{s['ttfa_p99']:>6.0f}ms | {s['total_avg']:>6.0f}ms | {s['rtf_avg']:>4.2f}x")
    print(f"{'='*70}\n")

    if audio_dir:
        wav_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
        print(f"Saved {len(wav_files)} audio file(s) to {audio_dir}/\n")

    _plot_ttfa_vs_concurrency(results_table)


def _plot_ttfa_vs_concurrency(results: list[dict], output_path: str = "ttfa_vs_concurrency.png") -> None:
    """Generate a TTFA vs concurrency chart and save it to disk."""
    if not HAS_MATPLOTLIB:
        print("Skipping chart (install matplotlib to enable): pip install matplotlib")
        return
    if not results:
        return

    concurrency = [s["concurrency"] for s in results]
    avg = [s["ttfa_avg"] for s in results]
    p50 = [s["ttfa_p50"] for s in results]
    p95 = [s["ttfa_p95"] for s in results]
    p99 = [s["ttfa_p99"] for s in results]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(concurrency, avg, "o-", label="Avg", linewidth=2, markersize=7)
    ax.plot(concurrency, p50, "s--", label="P50", linewidth=1.5, markersize=6)
    ax.plot(concurrency, p95, "^--", label="P95", linewidth=1.5, markersize=6)
    ax.plot(concurrency, p99, "D--", label="P99", linewidth=1.5, markersize=6, alpha=0.8)

    ax.fill_between(concurrency, p50, p95, alpha=0.12, label="P50–P95 band")

    ax.set_xlabel("Concurrency", fontsize=13)
    ax.set_ylabel("TTFA (ms)", fontsize=13)
    ax.set_title("Time to First Audio vs Concurrency", fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(concurrency)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Chart saved to {output_path}")


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

    # Session config options
    parser.add_argument("--model", default=None, help="Model name")
    parser.add_argument("--voice", default="Vivian", help="Speaker voice")
    parser.add_argument(
        "--task-type",
        default="Base",
        choices=["Base"],
        help="TTS task type",
    )
    parser.add_argument("--language", default="Auto", help="Language")
    parser.add_argument("--instructions", default=None, help="Voice style instructions")
    parser.add_argument(
        "--response-format",
        default="wav",
        choices=["wav", "pcm", "flac", "mp3", "aac", "opus"],
        help="Audio format",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed (0.25-4.0)")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Max tokens")

    # Voice cloning options
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

    # Benchmark mode
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run concurrency vs TTFA sweep benchmark",
    )
    parser.add_argument(
        "--concurrency-levels",
        type=str,
        default="1,2,4,8,16",
        help="Comma-separated concurrency levels to test (default: 1,2,4,8,16)",
    )
    parser.add_argument(
        "--requests-per-level",
        type=int,
        default=10,
        help="Number of requests per concurrency level (default: 10)",
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=2,
        help="Number of warmup requests before benchmark (default: 2)",
    )
    parser.add_argument(
        "--match-concurrency",
        action="store_true",
        help="Set requests per level equal to concurrency level (e.g., 64 requests at concurrency=64)",
    )
    parser.add_argument(
        "--save-audio",
        action="store_true",
        help="Save WAV output from each benchmark request to benchmark_audio/",
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

    # Build session config (only include non-None values)
    config: dict = {}
    for key in [
        "model",
        "voice",
        "task_type",
        "language",
        "instructions",
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

    # Benchmark mode
    if args.benchmark:
        if not args.text:
            print("Error: --text is required for benchmark mode")
            raise SystemExit(1)
        concurrency_levels = [int(x.strip()) for x in args.concurrency_levels.split(",")]
        asyncio.run(
            run_concurrency_sweep(
                url=args.url,
                text=args.text,
                config=config,
                concurrency_levels=concurrency_levels,
                requests_per_level=args.requests_per_level,
                warmup_requests=args.warmup_requests,
                save_audio=args.save_audio,
                match_concurrency=args.match_concurrency,
            )
        )
        return

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
