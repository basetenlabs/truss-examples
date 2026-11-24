"""Audio helpers: ffmpeg decode to 24 kHz mono PCM, WAV encode, duration probe.

``decode`` is the single-process decode every route used before. ``decode_long`` is the
chunked-request path: the same decode on a thread pool, or with VV_PREP_PARALLEL=1 for long
audio, integer-second segments in parallel ffmpeg processes (``-ss`` with a pre-roll that is
decoded and dropped), checking that adjacent segments agree on an overlap past every seam and
falling back to the sequential decode when any seam disagrees. See the README's audio
preparation lever.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import os
import struct
import subprocess
import tempfile
import wave
from concurrent.futures import Executor
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 24_000  # VibeVoice's native rate (plugin resamples to this anyway)


class AudioError(ValueError):
    """The audio could not be decoded (maps to HTTP 400)."""


class FetchError(Exception):
    """A remote audio URL could not be reached (maps to HTTP 502)."""


def wav_duration_s(data: bytes) -> float | None:
    """Duration of a RIFF/WAVE PCM file from its header, or None if not a parsable WAV."""
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None
    try:
        with wave.open(io.BytesIO(data)) as w:
            rate = w.getframerate()
            return w.getnframes() / rate if rate else None
    except (wave.Error, EOFError, struct.error):
        return None


def is_flac(data: bytes) -> bool:
    return data[:4] == b"fLaC"


async def decode(data: bytes, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode any ffmpeg-readable audio to mono int16 at ``sample_rate``.

    The input is written to a temp file first so containers whose index sits at
    the end (AAC .m4a with a trailing moov atom) can be seeked.
    """
    if not data:
        raise AudioError("empty audio")
    fd, path = tempfile.mkstemp(prefix="vv_in_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-i", path, "-ac", "1", "-ar", str(sample_rate), "-f", "s16le", "-",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )  # fmt: skip
        out, err = await proc.communicate()
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    if proc.returncode != 0:
        raise AudioError(f"ffmpeg decode failed: {err.decode(errors='replace')[:300]}")
    pcm = np.frombuffer(out, dtype=np.int16)
    if pcm.size == 0:
        raise AudioError("audio decoded to zero samples")
    return pcm


def to_wav(pcm: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(np.ascontiguousarray(pcm, dtype=np.int16).tobytes())
    return buf.getvalue()


def wav_header(n_samples: int, sample_rate: int = SAMPLE_RATE) -> bytes:
    """The 44-byte RIFF header ``wave`` writes for mono 16-bit PCM (``to_wav`` byte for byte)."""
    data = 2 * n_samples
    return struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + data, b"WAVE", b"fmt ", 16, 1, 1,
                       sample_rate, 2 * sample_rate, 2, 16, b"data", data)  # fmt: skip


WAV_URI_PREFIX = b"data:audio/wav;base64,"


def wav_b64(pcm: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """base64 of ``to_wav(pcm)`` as ASCII bytes, without the ``wave`` module's copies."""
    pcm = np.ascontiguousarray(pcm, dtype=np.int16)
    return base64.b64encode(wav_header(pcm.shape[0], sample_rate) + pcm.tobytes())


def wav_data_uri(pcm: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    return "data:audio/wav;base64," + base64.b64encode(to_wav(pcm, sample_rate)).decode("ascii")


def split_data_uri(uri: str) -> tuple[str, bytes]:
    """(mime, bytes) of a base64 data URI."""
    head, _, payload = uri.partition(",")
    if not head.startswith("data:") or ";base64" not in head:
        raise AudioError("only base64 data URIs are supported")
    mime = head[5:].split(";")[0]
    try:
        return mime, base64.b64decode(payload, validate=False)
    except (ValueError, TypeError) as exc:
        raise AudioError(f"bad base64 audio: {exc}") from exc


# --- chunked-request decode (VV_PREP_PARALLEL) -------------------------------------------------

_FF = ("ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error")


@dataclass
class DecodeInfo:
    mode: str = "seq"  # seq | parallel | fallback (seam mismatch -> sequential) | loose (mismatch kept)
    n_segments: int = 1
    probe_ms: float = 0.0
    decode_ms: float = 0.0
    seam_mismatch: int = 0


def _write_tmp(data: bytes) -> str:
    fd, path = tempfile.mkstemp(prefix="vv_in_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
    except BaseException:
        # Customer audio must not outlive the request (ZDR): drop a partial file.
        with contextlib.suppress(OSError):
            os.unlink(path)
        raise
    return path


def _unlink_written(fut: asyncio.Future) -> None:
    """Done-callback for a ``_write_tmp`` whose caller was cancelled: remove the file it wrote."""
    if not fut.cancelled() and fut.exception() is None:
        with contextlib.suppress(OSError):
            os.unlink(fut.result())


def probe_duration_s(path: str) -> float | None:
    """Container duration from ffprobe (reads the header/moov only), or None."""
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "csv=p=0", path], capture_output=True, timeout=30)  # fmt: skip
        return float(out.stdout.strip()) if out.returncode == 0 else None
    except (ValueError, OSError, subprocess.TimeoutExpired):
        return None


class _Cancel:
    """Lets the event loop kill a blocking ``_decode_file`` running in a pool thread."""

    def __init__(self):
        self.cancelled = False
        self.proc: subprocess.Popen | None = None

    def cancel(self) -> None:
        self.cancelled = True
        if self.proc is not None:
            with contextlib.suppress(OSError):
                self.proc.kill()


def _decode_file(path: str, sample_rate: int, ss: float = 0.0, t: float | None = None,
                 cancel: _Cancel | None = None) -> np.ndarray:
    """Blocking ffmpeg decode of ``path`` (optionally ``-ss ss -t t``) to mono int16."""
    cmd = list(_FF)
    if ss > 0:  # no -ss at 0: a seek to 0 skips the encoder priming differently from a plain decode
        cmd += ["-ss", f"{ss:.6f}"]
    cmd += ["-i", path]
    if t is not None:
        cmd += ["-t", f"{t:.6f}"]
    cmd += ["-ac", "1", "-ar", str(sample_rate), "-f", "s16le", "-"]
    if cancel is not None and cancel.cancelled:
        raise AudioError("decode cancelled")
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:
        if cancel is not None:
            cancel.proc = p
            if cancel.cancelled:
                p.kill()
        out, err = p.communicate()
    if p.returncode != 0:
        raise AudioError(f"ffmpeg decode failed: {err.decode(errors='replace')[:300]}")
    return np.frombuffer(out, dtype=np.int16)


def plan_segments(dur: float, n: int, min_seg_s: float) -> list[int]:
    """Integer-second segment starts (0 first). Integer seconds keep every segment on the same
    resampler phase as a whole-file decode for any integer input rate."""
    n = max(1, min(n, int(dur // max(1.0, min_seg_s))))
    seg = int(dur // n)
    return [i * seg for i in range(n)] if seg >= 1 else [0]


async def decode_long(data: bytes, pool: Executor, *, sample_rate: int = SAMPLE_RATE,
                      parallel: bool = True, min_parallel_s: float = 0.0, workers: int = 4,
                      min_seg_s: float = 120.0, preroll_s: float = 2.0, verify_s: float = 30.0,
                      fallback: bool = True) -> tuple[np.ndarray, DecodeInfo]:
    """Decode for a request that may be chunked; all blocking work runs on ``pool``.

    Sequential (identical to ``decode``) unless ``parallel`` and the probed duration exceeds
    ``min_parallel_s`` and ``2 * min_seg_s``. Segment ``k`` decodes from ``start_k - preroll_s``
    to ``start_{k+1} + verify_s``; the pre-roll is dropped and the ``verify_s`` overlap must equal
    the head of segment ``k+1`` sample for sample. Any mismatch (an inexact seek, or decoder state
    such as AAC PNS noise that a seek resets) makes the result sequential again when ``fallback``:
    the sequential decode then runs alongside the segments from the start (one more process; the
    pool needs ``workers + 1`` threads) and is killed as soon as every seam matches, so a fallback
    costs about what the sequential decode alone would.
    """
    if not data:
        raise AudioError("empty audio")
    loop = asyncio.get_running_loop()
    info = DecodeInfo()
    write = loop.run_in_executor(pool, _write_tmp, data)
    try:
        path = await asyncio.shield(write)
    except asyncio.CancelledError:
        # The write still finishes on the pool; delete its file then instead of leaking it.
        write.add_done_callback(_unlink_written)
        raise
    try:
        starts = [0]
        if parallel:
            t0 = loop.time()
            dur = await loop.run_in_executor(pool, probe_duration_s, path)
            info.probe_ms = (loop.time() - t0) * 1000
            if dur is not None and dur > min_parallel_s and dur >= 2 * min_seg_s:
                starts = plan_segments(dur, workers, min_seg_s)
        t0 = loop.time()
        if len(starts) == 1:
            pcm = await loop.run_in_executor(pool, _decode_file, path, sample_rate)
        else:
            info.n_segments = len(starts)
            cancel, seq = _Cancel(), None
            if fallback:
                seq = loop.run_in_executor(pool, _decode_file, path, sample_rate, 0.0, None, cancel)
            try:
                pcm, info.seam_mismatch = await _decode_segments(loop, pool, path, starts, sample_rate,
                                                                 preroll_s, verify_s)  # fmt: skip
            except BaseException:
                cancel.cancel()
                raise
            if not info.seam_mismatch:
                info.mode = "parallel"
                if seq is not None:
                    cancel.cancel()
                    with contextlib.suppress(AudioError):
                        await seq
            elif seq is not None:
                info.mode = "fallback"
                pcm = await seq
            else:
                info.mode = "loose"
        info.decode_ms = (loop.time() - t0) * 1000
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    if pcm.size == 0:
        raise AudioError("audio decoded to zero samples")
    return pcm, info


async def _decode_segments(loop, pool, path, starts, rate, preroll_s, verify_s) -> tuple[np.ndarray, int]:
    """(concatenated segments, number of seams whose overlap disagrees)."""
    jobs = []
    for k, s in enumerate(starts):
        ss = max(0.0, s - preroll_s)
        t = None if k == len(starts) - 1 else (starts[k + 1] - ss) + verify_s
        jobs.append(loop.run_in_executor(pool, _decode_file, path, rate, ss, t))
    outs = await asyncio.gather(*jobs)
    parts, mismatch = [], 0
    heads = [raw[int(round((s - max(0.0, s - preroll_s)) * rate)):] for s, raw in zip(starts, outs)]
    for k, x in enumerate(heads):
        if k < len(starts) - 1:
            keep = (starts[k + 1] - starts[k]) * rate
            tail, nxt = x[keep:], heads[k + 1]
            n = min(tail.shape[0], nxt.shape[0])
            if (x.shape[0] < keep or n < min(int(verify_s * rate), nxt.shape[0])
                    or not np.array_equal(tail[:n], nxt[:n])):  # fmt: skip
                mismatch += 1
            x = x[:keep]
        parts.append(x)
    return np.concatenate(parts), mismatch
