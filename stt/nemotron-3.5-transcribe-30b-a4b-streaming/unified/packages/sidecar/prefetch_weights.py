"""Read the served checkpoint into the page cache while the engine imports.

start_vllm.sh runs this when NEMOTRON_WEIGHT_PREFETCH=early. vLLM's own prefetch
(--safetensors-load-strategy=prefetch) starts only when weight loading begins, and reads each
file on one thread; the checkpoint here is one large shard, so that is one sequential reader.
This starts before `vllm serve` is exec'd, overlapping the ~3 minutes of engine imports, and
reads every shard in parallel ranges. The target load and the MTP draft's second pass over the
same shard then read from memory.

Two steps, so a bad setup fails the start instead of falling back to a cold load:

    prefetch_weights.py check DIR   # synchronous: shards exist, fit in memory
    prefetch_weights.py read DIR --kill-on-error PID   # background: reads, logs GiB/s

`read` sends SIGTERM to PID (the sidecar, which nemotron_stt's supervisor turns into a server
exit) if any read fails, so a broken prefetch never leaves the replica starting quietly.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import signal
import sys
import threading
import time

PREFIX = "prefetch_weights:"
# Leave room for the engine's own host memory (Python, CUDA context, pinned buffers).
MAX_FRACTION_OF_AVAILABLE = 0.8


def log(msg: str) -> None:
    print(f"{PREFIX} {msg}", file=sys.stderr, flush=True)


def shards(root: str) -> list[str]:
    found: dict[str, str] = {}
    for dirpath, _, files in os.walk(root, followlinks=True):
        for name in files:
            if name.endswith(".safetensors"):
                path = os.path.join(dirpath, name)
                found.setdefault(os.path.realpath(path), path)
    return sorted(found)


def _read_int(path: str) -> int | None:
    try:
        with open(path) as f:
            raw = f.read().strip()
    except OSError:
        return None
    return None if raw in ("", "max") else int(raw)


def available_bytes() -> tuple[int, str]:
    """Memory the page cache can grow into: MemAvailable, capped by the cgroup's headroom."""
    meminfo: dict[str, int] = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, _, rest = line.partition(":")
            meminfo[key] = int(rest.split()[0]) * 1024
    avail, source = meminfo["MemAvailable"], "MemAvailable"
    limit = _read_int("/sys/fs/cgroup/memory.max")
    used = _read_int("/sys/fs/cgroup/memory.current")
    if limit is None:  # cgroup v1
        limit = _read_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        used = _read_int("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    if limit is not None and limit < (1 << 60) and used is not None and limit - used < avail:
        avail, source = limit - used, "cgroup memory.max - memory.current"
    return avail, source


def check(root: str) -> list[str]:
    files = shards(root)
    if not files:
        log(f"ERROR no .safetensors under {root}")
        sys.exit(2)
    total = sum(os.path.getsize(p) for p in files)
    avail, source = available_bytes()
    gib = 1024**3
    log(
        f"{len(files)} shard(s), {total / gib:.2f} GiB; {source} {avail / gib:.2f} GiB "
        f"(limit {MAX_FRACTION_OF_AVAILABLE:.0%})"
    )
    if total > MAX_FRACTION_OF_AVAILABLE * avail:
        log(
            "ERROR the checkpoint does not fit in the page cache this container can hold; "
            "unset NEMOTRON_WEIGHT_PREFETCH or raise the memory, rather than prefetching into "
            "eviction"
        )
        sys.exit(3)
    return files


def read(root: str, threads: int, block: int, kill_pid: int | None) -> None:
    files = check(root)
    ranges = [(p, off) for p in files for off in range(0, os.path.getsize(p), block)]
    local = threading.local()

    def fd_for(path: str) -> int:
        fds = getattr(local, "fds", None)
        if fds is None:
            fds = local.fds = {}
            local.buf = bytearray(block)
        if path not in fds:
            fds[path] = os.open(path, os.O_RDONLY)
        return fds[path]

    def read_range(task: tuple[str, int]) -> int:
        path, off = task
        fd = fd_for(path)  # also allocates this thread's buffer
        return os.preadv(fd, [local.buf], off)

    start = time.monotonic()
    done = 0
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as pool:
            for n in pool.map(read_range, ranges):
                done += n
    except Exception as exc:  # any failure must stop the start, not just slow it
        log(f"ERROR read failed after {done / 1024**3:.2f} GiB: {type(exc).__name__}: {exc}")
        if kill_pid:
            log(f"stopping the sidecar (pid {kill_pid})")
            os.kill(kill_pid, signal.SIGTERM)
        sys.exit(1)
    elapsed = time.monotonic() - start
    log(
        f"read {done / 1024**3:.2f} GiB in {elapsed:.1f} s "
        f"({done / 1024**3 / max(elapsed, 1e-9):.2f} GiB/s, {threads} threads, "
        f"{block // (1024 * 1024)} MiB blocks)"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["check", "read"])
    parser.add_argument("root")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--block-mib", type=int, default=16)
    parser.add_argument("--kill-on-error", type=int, default=None)
    args = parser.parse_args()
    if args.threads < 1 or args.block_mib < 1:
        parser.error("--threads and --block-mib must be positive")
    if args.mode == "check":
        check(args.root)
    else:
        read(args.root, args.threads, args.block_mib * 1024 * 1024, args.kill_on_error)


if __name__ == "__main__":
    main()
