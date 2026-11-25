"""Server-side chunking of long audio for Nemotron 3.5 Transcribe: plan, decode, stitch.

The model card limits one request to 1 hour of audio (65,536-token window, trained
length). NVIDIA's own long-form evaluation splits files into 55-minute chunks with
5 seconds of overlap. ``chunk_proxy.py`` does the same on the server, so a client can
send a 2-hour call in one request:

    audio (2 h, over the 1-hour trigger) ──decode once, 16 kHz mono──▶ raw PCM + 20 ms frame energies
         │
         ├─ plan_chunks: cut at the quietest 200 ms in the last ``search`` seconds
         │  before each chunk's limit; the next chunk starts ``overlap`` seconds
         │  before the cut
         │       chunk 0 [0 ─────────── cut1)
         │       chunk 1          [cut1-5s ─────────── cut2)
         │       chunk 2                       [cut2-5s ──── end)
         │
         ├─ each chunk → a 16 kHz mono WAV the engine reads as a file:// URL
         │
         └─ stitch_texts: align the tail of chunk i with the head of chunk i+1 word
            by word (case and punctuation ignored, substitutions and extra or missing
            words allowed) and join at the middle of the overlap, so each overlapped
            word comes from the chunk where it is farther from the chunk's edge

The cut lands in a pause, so chunk i ends cleanly; chunk i+1 may start mid-word, but
its first ``overlap`` seconds are also the end of chunk i, and the join keeps chunk i's
words for the first half of the overlap. A last chunk under 20% of the chunk limit is
folded into the previous one (or the chunks are evened out), so no request carries a few
seconds of audio alone. The quiet-frame search is the Whisper unified backend's
``_windows`` (stt/whisper-unified-common/packages/whisper_stt/backend.py), scaled
from 28 s windows to 55-minute chunks.

This module has no web dependency, so the planning, decode and stitching are tested
without a server or an engine (tests/test_long_audio.py).
"""

from __future__ import annotations

import copy
import math
import os
import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

#: The model's input rate. Chunks are written at it, so the engine does not resample.
SAMPLE_RATE = 16_000
#: Energy frame for the quiet-cut search.
FRAME_S = 0.02
#: The cut is the middle of the quietest run of this many frames (200 ms).
SMOOTH_FRAMES = 10
#: Inline audio smaller than ``max_seconds * this`` bytes cannot be longer than
#: ``max_seconds`` at any real codec's bitrate (4 kbps; Opus bottoms out at 6), so
#: the proxy forwards it without decoding anything.
MIN_BYTES_PER_SECOND = 500
#: Words of each side compared when stitching, beyond those the overlap can hold.
STITCH_WORDS_PER_OVERLAP_S = 6
STITCH_EXTRA_WORDS = 8
#: Native-rate samples decoded before each downmix/resample/write step.
_DECODE_BATCH = 1 << 16


class AudioInputError(ValueError):
    """The request's audio cannot be served (too long, too big, undecodable): HTTP 400."""


@dataclass(frozen=True)
class ChunkConfig:
    """Chunking limits, read from the environment by :meth:`from_env`."""

    #: Chunks are at most this long. 0 disables chunking.
    max_seconds: float = 3300.0
    #: Only audio longer than this is chunked; up to it, one engine request (the model
    #: card's validated 1 hour, which BF16 serves cleanly unchunked).
    trigger_seconds: float = 3600.0
    #: Each chunk after the first starts this much before the previous chunk's cut.
    overlap_seconds: float = 5.0
    #: How far back from a chunk's limit the cut may move to find a pause.
    search_seconds: float = 30.0
    #: The longest audio one request may carry (decoded duration).
    max_audio_seconds: float = 10800.0
    #: The largest audio file one request may carry (downloaded or inline), in bytes.
    max_audio_bytes: int = 1536 * 1024 * 1024
    #: Files bigger than this go through the decode path even when short enough:
    #: the engine refuses them (VLLM_MAX_AUDIO_CLIP_FILESIZE_MB), a 16 kHz mono WAV it accepts.
    engine_max_file_bytes: int = 180 * 1024 * 1024

    @property
    def enabled(self) -> bool:
        return self.max_seconds > 0

    @property
    def inline_probe_min_bytes(self) -> int:
        return int(self.trigger_seconds * MIN_BYTES_PER_SECOND)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "ChunkConfig":
        """Read and validate the NEMOTRON_CHUNK_* knobs. Raises ValueError on any misconfiguration."""
        env = os.environ if env is None else env

        def number(name: str, default: float) -> float:
            raw = env.get(name, "")
            if raw.strip() == "":
                return default
            try:
                value = float(raw)
            except ValueError:
                raise ValueError(f"{name} must be a number of seconds, got {raw!r}") from None
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a non-negative number, got {raw!r}")
            return value

        cfg = cls(
            max_seconds=number("NEMOTRON_CHUNK_MAX_SECONDS", cls.max_seconds),
            trigger_seconds=number("NEMOTRON_CHUNK_TRIGGER_SECONDS", cls.trigger_seconds),
            overlap_seconds=number("NEMOTRON_CHUNK_OVERLAP_SECONDS", cls.overlap_seconds),
            search_seconds=number("NEMOTRON_CHUNK_SEARCH_SECONDS", cls.search_seconds),
            max_audio_seconds=number("NEMOTRON_MAX_AUDIO_SECONDS", cls.max_audio_seconds),
            max_audio_bytes=int(number("NEMOTRON_MAX_AUDIO_MB", cls.max_audio_bytes / 2**20) * 2**20),
            engine_max_file_bytes=int(
                number("VLLM_MAX_AUDIO_CLIP_FILESIZE_MB", 25) * 2**20
            ),
        )
        if not cfg.enabled:
            return cfg
        engine_cap = number("VLLM_MAX_AUDIO_DECODE_DURATION_S", 600)
        if cfg.trigger_seconds < cfg.max_seconds:
            raise ValueError(
                f"NEMOTRON_CHUNK_TRIGGER_SECONDS={cfg.trigger_seconds:g} is below "
                f"NEMOTRON_CHUNK_MAX_SECONDS={cfg.max_seconds:g}"
            )
        if cfg.trigger_seconds > engine_cap:
            raise ValueError(
                f"NEMOTRON_CHUNK_TRIGGER_SECONDS={cfg.trigger_seconds:g} exceeds the engine's decode cap "
                f"VLLM_MAX_AUDIO_DECODE_DURATION_S={engine_cap:g}: unchunked audio up to it would be refused"
            )
        if cfg.max_seconds < 30:
            raise ValueError(f"NEMOTRON_CHUNK_MAX_SECONDS must be 0 (off) or at least 30, got {cfg.max_seconds:g}")
        if cfg.overlap_seconds + cfg.search_seconds >= cfg.max_seconds / 2:
            raise ValueError(
                "NEMOTRON_CHUNK_OVERLAP_SECONDS + NEMOTRON_CHUNK_SEARCH_SECONDS must be under half of "
                f"NEMOTRON_CHUNK_MAX_SECONDS ({cfg.overlap_seconds:g} + {cfg.search_seconds:g} vs {cfg.max_seconds:g})"
            )
        if cfg.max_audio_seconds < cfg.trigger_seconds:
            raise ValueError(
                f"NEMOTRON_MAX_AUDIO_SECONDS={cfg.max_audio_seconds:g} is below NEMOTRON_CHUNK_TRIGGER_SECONDS={cfg.trigger_seconds:g}"
            )
        # The longest single engine input (unchunked audio up to the trigger, transcoded
        # when its file is too big) at 16 kHz mono int16 must fit the engine's file limit.
        chunk_bytes = cfg.trigger_seconds * SAMPLE_RATE * 2 + 44
        if chunk_bytes > cfg.engine_max_file_bytes:
            raise ValueError(
                f"a {cfg.trigger_seconds:g} s input is {chunk_bytes / 2**20:.0f} MiB as 16 kHz WAV, over "
                f"VLLM_MAX_AUDIO_CLIP_FILESIZE_MB={cfg.engine_max_file_bytes / 2**20:g}"
            )
        return cfg


# --- planning ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    index: int
    #: Sample range of the chunk's audio (16 kHz).
    start: int
    end: int
    #: The WAV the engine reads (set once written).
    path: str = ""

    @property
    def start_s(self) -> float:
        return self.start / SAMPLE_RATE

    @property
    def end_s(self) -> float:
        return self.end / SAMPLE_RATE


#: A last chunk shorter than this fraction of ``max_seconds`` is folded into the previous
#: chunk (or the chunks are rebalanced): a few seconds of audio alone in a request is where
#: the model is least reliable, and it adds a seam for nothing.
MIN_TAIL_FRACTION = 0.2


def plan_chunks(energy: np.ndarray, n_samples: int, cfg: ChunkConfig, rate: int = SAMPLE_RATE) -> list[tuple[int, int]]:
    """Sample spans covering ``n_samples``: one span up to ``cfg.trigger_seconds``, otherwise
    spans each at most ``cfg.max_seconds`` long.

    ``energy`` is the mean square of each ``FRAME_S`` frame. Each cut is the middle of the
    quietest ``SMOOTH_FRAMES`` run in the last ``cfg.search_seconds`` before the chunk's limit;
    the next chunk starts ``cfg.overlap_seconds`` before that cut. Audio no longer than
    ``trigger_seconds`` is one span.

    No tiny last chunk: when the greedy plan's last span is under ``MIN_TAIL_FRACTION`` of
    ``max_seconds``, it is merged into the previous span if that stays within
    ``trigger_seconds`` (the longest single request), else the same number of chunks is
    re-planned with evenly spread cuts.
    """
    limit = int(cfg.max_seconds * rate)
    overlap = int(cfg.overlap_seconds * rate)
    search = int(cfg.search_seconds * rate)
    frame = max(1, int(FRAME_S * rate))
    if limit <= overlap + search:
        raise ValueError("chunk limit must exceed overlap + search")
    single = int(cfg.trigger_seconds * rate)
    if n_samples <= single:
        return [(0, n_samples)]
    spans = _greedy(energy, n_samples, limit, overlap, search, frame)
    tail = spans[-1][1] - spans[-1][0]
    if len(spans) < 2 or tail >= MIN_TAIL_FRACTION * limit:
        return spans
    if n_samples - spans[-2][0] <= single:
        return spans[:-2] + [(spans[-2][0], n_samples)]
    even = _even(energy, n_samples, len(spans), limit, overlap, search, frame)
    return even if even is not None else spans


def _greedy(energy, n_samples, limit, overlap, search, frame) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    while n_samples - start > limit:
        hi = start + limit
        cut = _quietest(energy, hi - search, hi, frame)
        spans.append((start, cut))
        start = cut - overlap
    spans.append((start, n_samples))
    return spans


def _even(energy, n_samples, k, limit, overlap, search, frame) -> list[tuple[int, int]] | None:
    """``k`` spans of about equal length: each cut searched in the ``search`` window centred
    on that chunk's share of what is left. None if the plan would break ``limit``."""
    spans: list[tuple[int, int]] = []
    start = 0
    for left in range(k, 1, -1):
        share = (n_samples - start + (left - 1) * overlap) / left
        hi = start + min(limit, int(share + search / 2))
        cut = _quietest(energy, hi - search, hi, frame)
        spans.append((start, cut))
        start = cut - overlap
    spans.append((start, n_samples))
    if any(e - s > limit or e - s <= overlap for s, e in spans):
        return None
    return spans


def _quietest(energy: np.ndarray, lo: int, hi: int, frame: int) -> int:
    """Sample index in ``[lo, hi]`` at the middle of the quietest smoothed run of frames."""
    f_lo = -(-lo // frame)  # first whole frame at or after lo
    f_hi = hi // frame  # frames [f_lo, f_hi) end at or before hi
    region = energy[f_lo:f_hi]
    if len(region) < SMOOTH_FRAMES:
        return hi
    smoothed = np.convolve(region, np.ones(SMOOTH_FRAMES, dtype=np.float64), mode="valid")
    best = int(np.argmin(smoothed))
    return min(hi, (f_lo + best + SMOOTH_FRAMES // 2) * frame)


# --- decode -----------------------------------------------------------------------------


def probe_duration(path: str | Path) -> float | None:
    """Duration from container/stream metadata, without decoding; None if it has none."""
    import av

    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise AudioInputError("No audio stream found.")
            stream = container.streams.audio[0]
            if stream.duration and stream.time_base:
                return float(stream.duration * stream.time_base)
            if container.duration:
                return container.duration / 1_000_000
            return None
    except av.error.FFmpegError as e:
        raise AudioInputError(f"Invalid or corrupted audio data: {e}") from e


def decode_to_pcm(src: str | Path, raw_path: str | Path, max_seconds: float) -> tuple[int, np.ndarray]:
    """Decode ``src`` to 16 kHz mono int16 at ``raw_path``, streaming, and return
    ``(n_samples, frame_energy)``.

    Memory stays at a few decoder frames whatever the length: samples go to disk as
    they are produced, and only one float per 20 ms frame is kept (1.4 MB for 2 hours).
    Channels are averaged before resampling, as vLLM's own loader does.
    """
    import av

    limit = int(max_seconds * SAMPLE_RATE)
    frame_len = int(FRAME_S * SAMPLE_RATE)
    energies: list[np.ndarray] = []
    carry = np.zeros(0, dtype=np.float32)
    n = 0
    to_float = av.AudioResampler(format="fltp")  # native rate and layout, planar float
    to_16k: av.AudioResampler | None = None
    native_rate = 0

    def emit(mono: np.ndarray, fh) -> None:
        nonlocal carry, n
        n += len(mono)
        if n > limit:
            raise AudioInputError(
                f"Audio is longer than the {max_seconds:g} s this deployment accepts per request "
                "(NEMOTRON_MAX_AUDIO_SECONDS)."
            )
        np.clip(np.rint(mono * 32768.0), -32768, 32767).astype("<i2").tofile(fh)
        buf = np.concatenate([carry, mono]) if len(carry) else mono
        whole = len(buf) // frame_len * frame_len
        if whole:
            energies.append((buf[:whole].reshape(-1, frame_len).astype(np.float64) ** 2).mean(axis=1).astype(np.float32))
        carry = buf[whole:].astype(np.float32, copy=True)

    def resampled(mono: np.ndarray | None) -> list[np.ndarray]:
        if to_16k is None:
            return [] if mono is None else [mono]
        if mono is None:
            frames = to_16k.resample(None)
        else:
            f = av.AudioFrame.from_ndarray(mono[None, :], format="fltp", layout="mono")
            f.sample_rate = native_rate
            frames = to_16k.resample(f)
        return [fr.to_ndarray()[0].astype(np.float32, copy=False) for fr in frames]

    try:
        with av.open(str(src)) as container, open(raw_path, "wb") as fh:
            if not container.streams.audio:
                raise AudioInputError("No audio stream found.")
            stream = container.streams.audio[0]
            stream.thread_type = "AUTO"
            # Decoder frames are ~1,024 samples; per-frame Python work cost 10x the decode
            # itself, so frames are batched to ~64k samples before downmix and resampling.
            pending: list[np.ndarray] = []
            pending_n = 0

            def flush() -> None:
                nonlocal pending, pending_n
                if pending:
                    mono = np.concatenate(pending, axis=1).mean(axis=0, dtype=np.float32)
                    pending, pending_n = [], 0
                    for out in resampled(mono):
                        emit(out, fh)

            for frame in container.decode(stream):
                if not native_rate:
                    native_rate = frame.sample_rate
                    if native_rate != SAMPLE_RATE:
                        to_16k = av.AudioResampler(format="fltp", layout="mono", rate=SAMPLE_RATE)
                planars = [frame] if frame.format.name == "fltp" else to_float.resample(frame)
                for planar in planars:
                    arr = planar.to_ndarray()
                    pending.append(arr)
                    pending_n += arr.shape[-1]
                if pending_n >= _DECODE_BATCH:
                    flush()
            flush()
            for out in resampled(None):
                emit(out, fh)
    except av.error.FFmpegError as e:
        raise AudioInputError(f"Invalid or corrupted audio data: {e}") from e
    if n == 0:
        raise AudioInputError("No audio found in the input.")
    energy = np.concatenate(energies) if energies else np.zeros(0, dtype=np.float32)
    return n, energy


def write_wav(raw_path: str | Path, start: int, end: int, out_path: str | Path, block: int = 1 << 22) -> None:
    """Copy samples ``[start, end)`` of the raw int16 file into a 16 kHz mono WAV, in blocks."""
    pcm = np.memmap(raw_path, dtype="<i2", mode="r")
    try:
        with wave.open(str(out_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            for i in range(start, end, block):
                w.writeframes(pcm[i : min(end, i + block)].tobytes())
    finally:
        del pcm


def decode_and_split(src: str, out_dir: str, cfg: ChunkConfig) -> list[Chunk]:
    """Decode ``src`` once and write its chunks as WAVs in ``out_dir``. Runs in a worker process."""
    raw = Path(out_dir) / "audio.s16"
    try:
        n, energy = decode_to_pcm(src, raw, cfg.max_audio_seconds)
        spans = plan_chunks(energy, n, cfg) if cfg.enabled else [(0, n)]
        chunks = []
        for i, (s, e) in enumerate(spans):
            path = Path(out_dir) / f"chunk{i:03d}.wav"
            write_wav(raw, s, e, path)
            chunks.append(Chunk(index=i, start=s, end=e, path=str(path)))
        return chunks
    finally:
        raw.unlink(missing_ok=True)


def split_in_two(src: str, out_dir: str, cfg: ChunkConfig, offset: int = 0) -> list[Chunk]:
    """Decode ``src`` and write it as two WAVs cut at the quietest 200 ms near its middle,
    the second starting ``cfg.overlap_seconds`` before the cut. Runs in a worker process.

    The runaway guard's retry (runaway.py): a unit whose transcript ran away is
    transcribed again as two halves. ``offset`` (samples) is where ``src`` starts in the
    request's audio, so the halves' spans are on the request's clock.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    raw = Path(out_dir) / "audio.s16"
    try:
        n, energy = decode_to_pcm(src, raw, cfg.max_audio_seconds)
        frame = max(1, int(FRAME_S * SAMPLE_RATE))
        overlap = int(cfg.overlap_seconds * SAMPLE_RATE)
        # Search the middle quarter at most (and at most ``search_seconds``).
        window = min(int(cfg.search_seconds * SAMPLE_RATE), n // 4)
        mid = n // 2
        cut = _quietest(energy, mid - window // 2, mid + window // 2, frame)
        spans = [(0, cut), (max(0, cut - overlap), n)]
        chunks = []
        for i, (s, e) in enumerate(spans):
            path = Path(out_dir) / f"half{i}.wav"
            write_wav(raw, s, e, path)
            chunks.append(Chunk(index=i, start=offset + s, end=offset + e, path=str(path)))
        return chunks
    finally:
        raw.unlink(missing_ok=True)


# --- stitching --------------------------------------------------------------------------

_NORM = re.compile(r"[^\w<>:']+", re.UNICODE)


def _norm(token: str) -> str:
    return _NORM.sub("", token.lower()).strip("'")


@dataclass(frozen=True)
class Seam:
    #: Overlap words both chunks transcribed the same (case and punctuation ignored);
    #: 0 means no alignment was found and the texts were concatenated as they are.
    matched: int
    #: Words of the earlier chunk dropped: its overlap words after the cut point (and any
    #: extra word it added there). The later chunk's copy of those words is kept.
    dropped_prev: int
    #: Words of the later chunk dropped: its overlap words before the cut point (including
    #: a word fragment it began with). The earlier chunk's copy is kept.
    dropped_next: int


# Alignment scores: a matched word outweighs a substitution or a gap on either side.
_MATCH, _MISMATCH, _GAP = 2, -1, -1


#: An alignment of 2+ matched words is trusted only at this score (two clean matches and
#: one slip): "plan ... is" matched across unrelated texts scores less.
_MIN_SCORE = 3


def _align_overlap(a: list[str], b: list[str]) -> tuple[list[tuple[int, int]], int]:
    """Index pairs ``(i, j)`` of equal words in the best overlap alignment of ``a`` (the
    earlier chunk's last words) and ``b`` (the later chunk's first words).

    Semi-global: the alignment must run to the end of ``a`` (the earlier chunk ends where
    the overlap ends) and start at the beginning of ``b`` (the later chunk starts where the
    overlap starts), while ``a``'s head and ``b``'s rest are free. A substitution, an extra
    or a missing word costs a little; a run of equal words far from where the overlap must
    be (a repeated "of the") cannot win, because reaching it costs a gap per skipped word.
    Empty normalized words (punctuation) never match. Returns ``(pairs, score)``.
    """
    n, m = len(a), len(b)
    score = [[0] * (m + 1) for _ in range(n + 1)]
    move = [[0] * (m + 1) for _ in range(n + 1)]  # 1 diagonal, 2 up (gap in b), 3 left (gap in a)
    for j in range(1, m + 1):
        # b's first word is free to skip: the later chunk may begin with a word fragment.
        score[0][j] = (j - 1) * _GAP
        move[0][j] = 3
    for i in range(1, n + 1):
        ai = a[i - 1]
        row, prev_row, mrow = score[i], score[i - 1], move[i]
        # a's last word is free to skip: the earlier chunk may end on a cut-off or extra word.
        a_gap = 0 if i == n else _GAP
        for j in range(1, m + 1):
            diag = prev_row[j - 1] + (_MATCH if ai and ai == b[j - 1] else _MISMATCH)
            up = prev_row[j] + a_gap
            left = row[j - 1] + (0 if j == 1 else _GAP)
            best, mv = diag, 1
            if up > best:
                best, mv = up, 2
            if left > best:
                best, mv = left, 3
            row[j], mrow[j] = best, mv
    end_j = max(range(m + 1), key=lambda j: (score[n][j], -j))
    if score[n][end_j] <= 0:
        return [], score[n][end_j]
    pairs = []
    i, j = n, end_j
    while i > 0 and j > 0:
        mv = move[i][j]
        if mv == 1:
            if a[i - 1] and a[i - 1] == b[j - 1]:
                pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif mv == 2:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs, score[n][end_j]


def stitch_pair(prev: str, nxt: str, overlap_seconds: float) -> tuple[str, Seam]:
    """Join two consecutive chunk transcripts, keeping each overlapped word once.

    The last ``window`` words of ``prev`` are aligned with the first ``window`` of ``nxt``
    (:func:`_align_overlap`, case and punctuation ignored). The join is at the middle
    matched word of that alignment: before it, ``prev``'s words are kept (they are far
    from ``prev``'s end, while ``nxt`` may have begun mid-word there); from it, ``nxt``'s
    (far from ``nxt``'s start, while ``prev``'s last words were near its cut). An
    alignment needs 2 matched words, or 1 when it is ``prev``'s last word and ``nxt``'s
    first (or second, after a fragment). Otherwise the overlap held no speech both chunks agree on (usually
    silence) and the texts are concatenated.
    """
    p_tok, n_tok = prev.split(), nxt.split()
    if not p_tok or not n_tok:
        return " ".join(p_tok + n_tok), Seam(0, 0, 0)
    window = int(math.ceil(overlap_seconds * STITCH_WORDS_PER_OVERLAP_S)) + STITCH_EXTRA_WORDS
    tail0 = max(0, len(p_tok) - window)
    a = [_norm(t) for t in p_tok[tail0:]]
    b = [_norm(t) for t in n_tok[:window]]
    pairs, score = _align_overlap(a, b)
    # One shared word is trusted only right at the edge: prev's last word and next's first
    # (or second, after a fragment).
    edge = len(pairs) == 1 and pairs[0][0] == len(a) - 1 and pairs[0][1] <= 1
    if (len(pairs) >= 2 and score >= _MIN_SCORE) or edge:
        i, j = pairs[len(pairs) // 2]
        keep_prev = tail0 + i
        seam = Seam(len(pairs), len(p_tok) - keep_prev, j)
        return " ".join(p_tok[:keep_prev] + n_tok[j:]), seam
    return " ".join(p_tok + n_tok), Seam(0, 0, 0)


def stitch_texts(texts: Sequence[str], overlap_seconds: float) -> tuple[str, list[Seam]]:
    if not texts:
        return "", []
    out = texts[0].strip()
    seams = []
    for t in texts[1:]:
        if overlap_seconds > 0:
            out, seam = stitch_pair(out, t.strip(), overlap_seconds)
        else:
            out, seam = " ".join(x for x in (out, t.strip()) if x), Seam(0, 0, 0)
        seams.append(seam)
    return out, seams


def snippet(text: str, chars: int = 200) -> tuple[str, str]:
    """The first and last ``chars`` characters of ``text`` (for logs)."""
    text = " ".join(text.split())
    return text[:chars], text[-chars:] if len(text) > chars else ""


# --- response merging -------------------------------------------------------------------


def _sum_usage(a: Any, b: Any) -> Any:
    if isinstance(a, bool) or isinstance(b, bool):
        return a
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a + b
    if isinstance(a, dict) and isinstance(b, dict):
        return {k: _sum_usage(a[k], b[k]) if k in a and k in b else a.get(k, b.get(k)) for k in {**a, **b}}
    if a is None:
        return b
    return a


def merge_finish_reason(reasons: Sequence[str | None]) -> str | None:
    """``length`` if any chunk was cut off, else the last chunk's reason."""
    if any(r == "length" for r in reasons):
        return "length"
    return reasons[-1] if reasons else None


def _merge_ctc(per_chunk: Sequence[dict | None], chunks: Sequence[Chunk]) -> dict | None:
    """Word timestamps and diarization moved onto the whole file's clock.

    Chunk i owns ``[its start, its cut)`` except that chunk i+1 owns from chunk i's cut on
    (the cut is in a pause, so no word straddles it). Speaker labels are per chunk: the same
    person can carry different labels in different chunks.
    """
    if all(c is None for c in per_chunk):
        return None
    words: list[dict] = []
    diar: list[dict] = []
    errors: list[str] = []
    for i, (res, ch) in enumerate(zip(per_chunk, chunks)):
        if not res:
            errors.append(f"chunk {i}: no timestamps")
            continue
        off = ch.start_s
        own_lo = 0.0 if i == 0 else chunks[i - 1].end_s
        own_hi = math.inf if i + 1 == len(chunks) else ch.end_s
        for w in res.get("words") or []:
            s = float(w.get("start", 0.0)) + off
            if own_lo <= s < own_hi:
                words.append({**w, "start": round(s, 3), "end": round(float(w.get("end", 0.0)) + off, 3)})
        for seg in res.get("diarization") or []:
            s = max(float(seg.get("start", 0.0)) + off, own_lo)
            e = min(float(seg.get("end", 0.0)) + off, own_hi)
            if e > s:
                diar.append({**seg, "start": round(s, 3), "end": round(e, 3)})
        if res.get("error"):
            errors.append(f"chunk {i}: {res['error']}")
    return {
        "words": words,
        "diarization": diar,
        "speaker_tag_to_diarization_speaker": {},
        "error": "; ".join(errors) or None,
        "chunks": [{"start": round(c.start_s, 3), "end": round(c.end_s, 3)} for c in chunks],
        "note": "audio was transcribed in chunks; speaker labels are per chunk",
    }


def merge_chat_responses(
    responses: Sequence[dict], chunks: Sequence[Chunk], overlap_seconds: float
) -> tuple[dict, list[Seam]]:
    """One chat completion from the chunks' completions, in the engine's own format."""
    if len(responses) != len(chunks) or not responses:
        raise ValueError("one response per chunk")
    merged = copy.deepcopy(responses[0])
    messages = [r["choices"][0].get("message") or {} for r in responses]
    text, seams = stitch_texts([m.get("content") or "" for m in messages], overlap_seconds)
    choice = merged["choices"][0]
    choice["message"]["content"] = text
    for key in ("reasoning_content", "reasoning"):
        parts = [m.get(key) for m in messages if m.get(key)]
        if key in choice["message"] or parts:
            choice["message"][key] = "\n".join(parts) if parts else choice["message"].get(key)
    choice["finish_reason"] = merge_finish_reason([r["choices"][0].get("finish_reason") for r in responses])
    usage = responses[0].get("usage")
    for r in responses[1:]:
        usage = _sum_usage(usage, r.get("usage"))
    if usage is not None:
        merged["usage"] = usage
    if any("ctc_timestamps" in r for r in responses):
        merged["ctc_timestamps"] = _merge_ctc([r.get("ctc_timestamps") for r in responses], chunks)
    return merged, seams
