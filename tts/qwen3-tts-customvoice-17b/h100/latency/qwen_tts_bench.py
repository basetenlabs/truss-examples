"""Closed-loop concurrency sweep for Qwen3-TTS CustomVoice on vLLM-Omni's streaming WebSocket.

At each level, N concurrent "agents" synthesize utterances back to back for --duration
seconds. Each agent keeps one WebSocket open for the whole level, sends session.config
once, then loops input.text + input.done (a flush, not a close), like a voice agent that
holds its TTS socket for the length of a call. Load is spread over --procs worker
processes, and every agent starts at the same instant.

Per utterance:
  connect = WebSocket open (once per agent; reported as the per-agent median)
  ttfa    = input.done sent -> first binary audio frame
  e2e     = input.done sent -> session.done
  rtf     = e2e / synthesized audio seconds (lower is better; < 1 = faster than real time)
  stall   = worst playback gap if audio starts playing at the first chunk: how far any chunk
            arrived after the moment it was due to play (0 = gapless playback)

Through Baseten:
    python3 qwen_tts_bench.py --url wss://model-<ID>.api.baseten.co/deployment/<DEP>/websocket \
        --api-key $KEY --concurrencies 1,2,4,8,16,32,64

--sample-cmd runs a shell command every 2 s while a level is running; it must print
"<gpu_util_pct> <cgroup_cpu_usage_usec>" (e.g. via kubectl exec into the pod).
"""

import argparse
import asyncio
import json
import multiprocessing as mp
import random
import resource
import statistics
import subprocess
import threading
import time

import websockets

SAMPLE_RATE = 24000

TEXTS = [
    "Sure, I can help with that.",
    "Your appointment is confirmed for Tuesday at three thirty in the afternoon.",
    "I'm sorry, I didn't catch that. Could you repeat the last four digits of your account number?",
    "Thanks for waiting. I found your order, and it looks like it shipped yesterday from our Denver warehouse.",
    "Of course. The total comes to forty two dollars and nineteen cents, including tax.",
    "Let me check on that for you. One moment please.",
    "Your new plan includes unlimited talk and text, plus fifteen gigabytes of high speed data every month.",
    "Great, you're all set. Is there anything else I can help you with today?",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="ws://localhost:8091/v1/audio/speech/stream")
    p.add_argument("--api-key", default=None)
    p.add_argument("--voice", default="vivian")
    p.add_argument("--instructions", default=None)
    p.add_argument("--language", default=None)
    p.add_argument("--concurrencies", default="1,2,4,8,16,32,64")
    p.add_argument("--duration", type=float, default=60.0, help="Seconds of closed-loop load per level.")
    p.add_argument("--procs", type=int, default=8)
    p.add_argument("--settle-s", type=float, default=5.0)
    p.add_argument("--timeout", type=float, default=120.0, help="Per-utterance timeout.")
    p.add_argument("--sample-cmd", default=None)
    p.add_argument("--json-out", default="tts_sweep.json")
    return p.parse_args()


def session_config(cfg):
    msg = {"type": "session.config", "task_type": "CustomVoice", "voice": cfg["voice"],
           "response_format": "pcm", "stream_audio": True}
    if cfg["language"]:
        msg["language"] = cfg["language"]
    if cfg["instructions"]:
        msg["instructions"] = cfg["instructions"]
    return msg


async def connect(cfg):
    headers = {"Authorization": f"Api-Key {cfg['api_key']}"} if cfg["api_key"] else {}
    t0 = time.perf_counter()
    ws = await websockets.connect(cfg["url"], additional_headers=headers, max_size=None,
                                  open_timeout=cfg["timeout"])
    await ws.send(json.dumps(session_config(cfg)))
    return ws, time.perf_counter() - t0


async def one_utterance(ws, text):
    await ws.send(json.dumps({"type": "input.text", "text": text}))
    await ws.send(json.dumps({"type": "input.done"}))
    t_sent = time.perf_counter()
    first = None
    nbytes = 0
    stall = 0.0
    while True:
        msg = await ws.recv()
        if isinstance(msg, bytes):
            now = time.perf_counter()
            if first is None:
                first = now
            # This chunk is due to start playing at first + audio already received.
            stall = max(stall, (now - first) - nbytes / 2 / SAMPLE_RATE)
            nbytes += len(msg)
            continue
        ev = json.loads(msg)
        if ev.get("type") == "session.done":
            break
        if ev.get("type") == "error":
            raise RuntimeError(f"synth: {ev}")
    t_done = time.perf_counter()
    audio_s = nbytes / 2 / SAMPLE_RATE
    if first is None or audio_s == 0:
        raise RuntimeError("no audio")
    e2e = t_done - t_sent
    return {"ttfa": first - t_sent, "e2e": e2e, "audio_s": audio_s, "rtf": e2e / audio_s,
            "stall": stall}


async def agent(cfg, start_at, end_at, idx, out):
    rng = random.Random(idx)
    ws = None
    try:
        while time.time() < end_at:
            try:
                if ws is None:
                    ws, conn = await asyncio.wait_for(connect(cfg), cfg["timeout"])
                    out["connect"].append(conn)
                    await asyncio.sleep(max(0.0, start_at - time.time()))
                r = await asyncio.wait_for(one_utterance(ws, rng.choice(TEXTS)), cfg["timeout"])
                r["end"] = time.time()
                out["ok"].append(r)
            except Exception as e:  # noqa: BLE001
                out["err"].append(repr(e)[:300])
                if ws is not None:
                    await ws.close()
                ws = None
                await asyncio.sleep(0.5)
    finally:
        if ws is not None:
            await ws.close()


def worker(cfg, n, start_at, end_at, base_idx, result_q):
    try:
        import uvloop
        uvloop.install()
    except ImportError:
        pass
    out = {"ok": [], "err": [], "connect": []}
    lag = [0.0]

    async def lag_probe():
        while time.time() < end_at + cfg["timeout"]:
            t = time.perf_counter()
            await asyncio.sleep(0.05)
            lag[0] = max(lag[0], time.perf_counter() - t - 0.05)

    async def main():
        probe = asyncio.create_task(lag_probe())
        await asyncio.gather(*(agent(cfg, start_at, end_at, base_idx + i, out) for i in range(n)))
        probe.cancel()

    c0 = time.process_time()
    w0 = time.time()
    asyncio.run(main())
    out["loop_lag"] = lag[0]
    out["wcpu"] = (time.process_time() - c0) / max(1e-6, time.time() - w0)
    result_q.put(out)


class Sampler:
    def __init__(self, cmd):
        self.cmd, self.gpu, self.cpu = cmd, [], []
        self._stop = threading.Event()
        self._t = None

    def _run(self):
        last = None
        while not self._stop.is_set():
            try:
                g, c = subprocess.run(self.cmd, shell=True, capture_output=True, text=True,
                                      timeout=10).stdout.split()[:2]
                now = (time.time(), int(c) / 1e6)
                self.gpu.append(float(g))
                if last:
                    self.cpu.append((now[1] - last[1]) / (now[0] - last[0]))
                last = now
            except Exception:  # noqa: BLE001
                pass
            self._stop.wait(2.0)

    def __enter__(self):
        if self.cmd:
            self._t = threading.Thread(target=self._run, daemon=True)
            self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self._t:
            self._t.join(15)


def pct(xs, q):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q / 100 * (len(xs) - 1))))]


def run_level(cfg, n, procs):
    procs = max(1, min(procs, n))
    per = [n // procs + (1 if i < n % procs else 0) for i in range(procs)]
    start_at = time.time() + 2.0
    end_at = start_at + cfg["duration"]
    q = mp.Queue()
    ps, base = [], 0
    for k in per:
        p = mp.Process(target=worker, args=(cfg, k, start_at, end_at, base, q))
        p.start()
        ps.append(p)
        base += k
    with Sampler(cfg["sample_cmd"]) as s:
        outs = [q.get() for _ in ps]
    for p in ps:
        p.join()
    ok = [r for o in outs for r in o["ok"]]
    err = [e for o in outs for e in o["err"]]
    # Count audio only from utterances that finished inside the load window, so a long
    # drain after end_at does not inflate throughput.
    window = [r for r in ok if r["end"] <= end_at + 1.0]
    return {
        "conc": n, "utts": len(ok), "errors": len(err), "first_error": err[0] if err else None,
        "connect_p50": pct([c for o in outs for c in o["connect"]], 50),
        "ttfa_p50": pct([r["ttfa"] for r in ok], 50), "ttfa_p95": pct([r["ttfa"] for r in ok], 95),
        "e2e_p50": pct([r["e2e"] for r in ok], 50),
        "rtf_p50": pct([r["rtf"] for r in ok], 50), "rtf_p95": pct([r["rtf"] for r in ok], 95),
        "stall_p95": pct([r["stall"] for r in ok], 95),
        "stalled_pct": 100 * sum(r["stall"] > 0.1 for r in ok) / max(1, len(ok)),
        "audio_x": sum(r["audio_s"] for r in window) / cfg["duration"],
        "gpu_avg": statistics.mean(s.gpu) if s.gpu else None,
        "gpu_max": max(s.gpu) if s.gpu else None,
        "cpu_avg": statistics.mean(s.cpu) if s.cpu else None,
        "loop_lag": max(o["loop_lag"] for o in outs), "wcpu": max(o["wcpu"] for o in outs),
    }


def fmt(v, spec, dash="-"):
    return dash if v is None or v != v else format(v, spec)


def main():
    args = parse_args()
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, 65536), hard))
    cfg = {"url": args.url, "api_key": args.api_key, "voice": args.voice,
           "language": args.language, "instructions": args.instructions, "duration": args.duration,
           "timeout": args.timeout, "sample_cmd": args.sample_cmd}
    print(f"url={args.url} voice={args.voice} duration={args.duration:.0f}s/level "
          f"procs={args.procs} texts={len(TEXTS)}")
    async def warm():
        ws, conn = await connect(cfg)
        try:
            r = await one_utterance(ws, TEXTS[1])
        finally:
            await ws.close()
        r["connect"] = conn
        return r

    w = asyncio.run(warm())
    print(f"warmup: connect={w['connect']:.2f}s ttfa={w['ttfa']:.3f}s e2e={w['e2e']:.2f}s "
          f"audio={w['audio_s']:.2f}s rtf={w['rtf']:.3f}\n")
    hdr = (" conc |  utts | err | connect | ttfaP50 | ttfaP95 |  e2eP50 | rtfP50 | rtfP95 | stallP95 "
           "| stalled | audioX | gpuAvg | gpuMax | cpuAvg | loopLag | wCPU")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for n in [int(x) for x in args.concurrencies.split(",")]:
        r = run_level(cfg, n, args.procs)
        rows.append(r)
        print(f" {n:4d} | {r['utts']:5d} | {r['errors']:3d} | {fmt(r['connect_p50'], '6.2f')}s "
              f"| {fmt(r['ttfa_p50'], '6.3f')}s | {fmt(r['ttfa_p95'], '6.3f')}s | {fmt(r['e2e_p50'], '6.2f')}s "
              f"| {fmt(r['rtf_p50'], '6.3f')} | {fmt(r['rtf_p95'], '6.3f')} | {fmt(r['stall_p95'], '7.3f')}s "
              f"| {fmt(r['stalled_pct'], '6.1f')}% | {fmt(r['audio_x'], '5.1f')}x "
              f"| {fmt(r['gpu_avg'], '5.1f'):>6} | {fmt(r['gpu_max'], '5.1f'):>6} | {fmt(r['cpu_avg'], '5.2f'):>6} "
              f"| {r['loop_lag']:6.2f}s | {r['wcpu']:.2f}", flush=True)
        if r["first_error"]:
            print(f"        first error @{n}: {r['first_error']}")
        with open(args.json_out, "w") as f:
            json.dump({"args": vars(args), "rows": rows}, f, indent=1)
        time.sleep(args.settle_s)
    print("""
connect = socket open + session.config (once per agent); ttfa = input.done -> first audio frame
e2e     = input.done -> session.done; rtf = e2e / audio seconds (< 1 = faster than real time)
stall   = worst playback gap per utterance if playback starts at the first chunk;
stalled = share of utterances with any gap > 100 ms
audioX  = audio seconds synthesized per wall second (x real time) within the load window
gpu/cpu = pod nvidia-smi util % and container cores (only with --sample-cmd)""")
    print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
