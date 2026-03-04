"""Sweep concurrency levels and plot p50/p90/p99 TTFA curves (voice cloning).

Runs the streaming TTS benchmark at concurrency 2, 4, 6, …, max_concurrency,
collects TTFA percentiles, and saves a plot + CSV. Uses a cached clone voice
when --voice-name is set, or creates a clone from --ref-audio (and optionally
--ref-text) when provided (task_type Base).

Usage:
    # Clone voice by name (cached on server)
    python sweep_concurrency.py --voice-name my_voice

    # Create clone from reference audio (and optional transcript)
    python sweep_concurrency.py --ref-audio /path/to/sample.wav --ref-text "Transcript of the audio."
    python sweep_concurrency.py --ref-audio /path/to/sample.wav --ref-text "Transcript." --voice-name my_voice

    python sweep_concurrency.py --voice-name my_voice --max-concurrency 20 --step 2 --rounds 5
    python sweep_concurrency.py --voice-name my_voice --output-plot ttfa_sweep.png
"""

import argparse
import asyncio
import base64
import csv
import json
import math
import mimetypes
import os
import time

try:
    import websockets
except ImportError:
    print("pip install websockets")
    raise SystemExit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("pip install matplotlib")
    raise SystemExit(1)


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


async def run_one_session(url: str, text: str, config: dict) -> dict:
    headers = {}
    if os.getenv("BASETEN_API_KEY"):
        headers["Authorization"] = f"Api-Key {os.getenv('BASETEN_API_KEY')}"
    async with websockets.connect(url, additional_headers=headers or None) as ws:
        config_msg = {"type": "session.config", **config}
        t0 = time.perf_counter()
        await ws.send(json.dumps(config_msg))
        await ws.send(json.dumps({"type": "input.text", "text": text}))
        await ws.send(json.dumps({"type": "input.done"}))

        ttfa = None
        total_pcm_bytes = 0

        while True:
            message = await ws.recv()
            if isinstance(message, bytes):
                if ttfa is None:
                    ttfa = time.perf_counter() - t0
                total_pcm_bytes += len(message)
            else:
                msg = json.loads(message)
                if msg.get("type") == "session.done":
                    break
                elif msg.get("type") == "error":
                    return {"error": msg["message"]}

        t_total = time.perf_counter() - t0
        duration_s = total_pcm_bytes / (24000 * 2) if total_pcm_bytes else 0
        return {
            "ttfa_ms": ttfa * 1000 if ttfa else None,
            "total_ms": t_total * 1000,
            "audio_duration_s": duration_s,
        }


async def run_at_concurrency(
    url: str, text: str, config: dict, concurrency: int, rounds: int,
) -> list[dict]:
    all_results: list[dict] = []
    for _ in range(rounds):
        tasks = [run_one_session(url, text, config) for _ in range(concurrency)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                all_results.append({"error": str(r)})
            else:
                all_results.append(r)
    return all_results


def percentile(vals: list[float], p: float) -> float:
    if not vals:
        return float("nan")
    vals = sorted(vals)
    k = (len(vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return vals[int(k)]
    return vals[f] * (c - k) + vals[c] * (k - f)


async def sweep(args):
    config = {
        "task_type": "Base",
        "language": "Auto",
        "response_format": "wav",
        "speed": 1.0,
    }
    if args.voice_name:
        config["voice_name"] = args.voice_name
    if args.ref_audio:
        if not os.path.isfile(args.ref_audio):
            print(f"Error: reference audio file not found: {args.ref_audio}")
            raise SystemExit(1)
        config["ref_audio"] = _encode_audio_file(args.ref_audio)
        if args.ref_text:
            config["ref_text"] = args.ref_text
        # ref_audio path implies clone; task_type already Base
    if not args.voice_name and not args.ref_audio:
        config["voice"] = args.voice
        config["task_type"] = args.task_type

    text = args.text

    concurrencies = list(range(args.step, args.max_concurrency + 1, args.step))

    desc = f"task_type={config['task_type']}"
    if config.get("voice_name"):
        desc += f" voice_name={config['voice_name']}"
    elif config.get("ref_audio"):
        desc += " ref_audio (clone)"
    else:
        desc += f" voice={config.get('voice')}"
    print(f"Target: {args.url}")
    print(f"Text: {text!r}")
    print(f"Config: {desc}")
    print(f"Concurrencies: {concurrencies}")
    print(f"Rounds per level: {args.rounds}")
    print(f"Warmup rounds: {args.warmup}")
    print("=" * 70)

    if args.warmup > 0:
        print(f"\n  Warmup ({args.warmup} round(s) at c=1)...", end="", flush=True)
        await run_at_concurrency(args.url, text, config, 1, args.warmup)
        print(" done")

    rows: list[dict] = []

    for conc in concurrencies:
        print(f"\n  [c={conc:>2}] benchmarking ({args.rounds} rounds × {conc} sessions)...", end="", flush=True)
        results = await run_at_concurrency(args.url, text, config, conc, args.rounds)
        print(" done")

        ok = [r for r in results if "error" not in r]
        ttfas = [r["ttfa_ms"] for r in ok if r.get("ttfa_ms") is not None]
        totals = [r["total_ms"] for r in ok]
        audio_secs = [r["audio_duration_s"] for r in ok]
        rtfs = [r["total_ms"] / 1000 / r["audio_duration_s"]
                for r in ok if r.get("audio_duration_s", 0) > 0]
        errors = len(results) - len(ok)

        p50 = percentile(ttfas, 50)
        p90 = percentile(ttfas, 90)
        p99 = percentile(ttfas, 99)
        rtf_p50 = percentile(rtfs, 50)
        rtf_p90 = percentile(rtfs, 90)
        rtf_p99 = percentile(rtfs, 99)
        total_p50 = percentile(totals, 50)
        total_audio = sum(audio_secs)
        wall = max(totals) / 1000 if totals else 0
        throughput = total_audio / wall if wall > 0 else 0

        row = {
            "concurrency": conc,
            "samples": len(ttfas),
            "errors": errors,
            "ttfa_p50": p50,
            "ttfa_p90": p90,
            "ttfa_p99": p99,
            "rtf_p50": rtf_p50,
            "rtf_p90": rtf_p90,
            "rtf_p99": rtf_p99,
            "total_p50": total_p50,
            "throughput_x": throughput,
        }
        rows.append(row)

        print(
            f"           TTFA  p50={p50:>7.0f}ms  p90={p90:>7.0f}ms  p99={p99:>7.0f}ms  "
            f"| RTF  p50={rtf_p50:.2f}x  p90={rtf_p90:.2f}x  p99={rtf_p99:.2f}x  "
            f"| throughput={throughput:.1f}x"
            f"{f'  errors={errors}' if errors else ''}"
        )

    # --- Save CSV ---
    csv_path = args.output_csv
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved: {csv_path}")

    # --- Plot: TTFA and RTF ---
    concs = [r["concurrency"] for r in rows]
    p50s = [r["ttfa_p50"] for r in rows]
    p90s = [r["ttfa_p90"] for r in rows]
    p99s = [r["ttfa_p99"] for r in rows]
    rp50s = [r["rtf_p50"] for r in rows]
    rp90s = [r["rtf_p90"] for r in rows]
    rp99s = [r["rtf_p99"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(concs, p50s, "o-", label="p50", linewidth=2, markersize=6)
    ax1.plot(concs, p90s, "s-", label="p90", linewidth=2, markersize=6)
    ax1.plot(concs, p99s, "^-", label="p99", linewidth=2, markersize=6)
    ax1.set_xlabel("Concurrency", fontsize=12)
    ax1.set_ylabel("TTFA (ms)", fontsize=12)
    ax1.set_title("Time to First Audio vs Concurrency", fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(concs)

    ax2.plot(concs, rp50s, "o-", label="p50", linewidth=2, markersize=6)
    ax2.plot(concs, rp90s, "s-", label="p90", linewidth=2, markersize=6)
    ax2.plot(concs, rp99s, "^-", label="p99", linewidth=2, markersize=6)
    ax2.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="realtime")
    ax2.set_xlabel("Concurrency", fontsize=12)
    ax2.set_ylabel("RTF (lower = faster)", fontsize=12)
    ax2.set_title("Real-Time Factor vs Concurrency", fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(concs)

    fig.tight_layout()
    plot_path = args.output_plot
    os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    print(f"Plot saved: {plot_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Sweep concurrency & plot TTFA percentiles (voice cloning)")
    parser.add_argument("--url", default="wss://model-wgl5y7o3.api.baseten.co/deployment/q41xjd5/websocket")
    parser.add_argument(
        "--text",
        default="Hello world. How are you today? I am doing very well, thank you for asking.",
    )
    parser.add_argument("--max-concurrency", type=int, default=16)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=3, help="Measured rounds per concurrency level")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup rounds (discarded) per level")
    parser.add_argument(
        "--voice-name",
        default=None,
        help="Cached clone voice name (uses task_type Base). Optional when using --ref-audio to cache after first run.",
    )
    parser.add_argument(
        "--ref-audio",
        default=None,
        help="Path to reference audio file for voice cloning. When set with optional --ref-text, creates clone (task_type Base).",
    )
    parser.add_argument(
        "--ref-text",
        default=None,
        help="Transcript of reference audio (optional). Can be inline text or path to a .txt file.",
    )
    parser.add_argument(
        "--voice",
        default="Vivian",
        help="Built-in voice name (used only when --voice-name is not set)",
    )
    parser.add_argument(
        "--task-type",
        default="CustomVoice",
        help="Task type when not using --voice-name (e.g. CustomVoice)",
    )
    parser.add_argument("--output-plot", default="ttfa_sweep.png")
    parser.add_argument("--output-csv", default="ttfa_sweep.csv")
    args = parser.parse_args()

    # If ref-text looks like a file path, read its contents
    if args.ref_text and os.path.isfile(args.ref_text):
        with open(args.ref_text) as f:
            args.ref_text = f.read().strip()

    if not args.voice_name and not args.ref_audio:
        print("Note: No --voice-name or --ref-audio set. Using built-in voice (not cloning). Use --voice-name NAME or --ref-audio PATH for clone sweep.")
    asyncio.run(sweep(args))


if __name__ == "__main__":
    main()