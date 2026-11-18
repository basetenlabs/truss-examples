"""Audio decode for the transcriptions route: torchcodec in one native call.

Installed as ``vllm/b10_decode.py`` by ``patch.py``; the patched
``vllm.multimodal.media.audio.load_audio`` routes every upload here instead of the
soundfile -> PyAV chain. ``torchcodec.decoders.AudioDecoder`` (FFmpeg inside libtorchcodec)
decodes, downmixes and resamples in one call, so the per-frame Python loop of
``load_audio_pyav`` never runs. With ``B10_DECODE_SEGMENTS=N>1``, files longer than
``N x B10_DECODE_SEGMENT_TARGET_S`` are decoded as up to N parallel time ranges spliced on
the sample grid.

The duration and byte bounds mirror the vLLM originals so the same inputs are rejected
with the same error class (``ValueError``).
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _read_bytes(path: BytesIO | Path | str) -> bytes:
    if isinstance(path, BytesIO):
        return path.getvalue()
    with open(path, "rb") as f:
        return f.read()


def _check_bounds(
    num_samples: int,
    num_channels: int,
    sr: float,
    max_duration_s: float | None,
    max_decode_bytes: int | None,
) -> None:
    duration_s = num_samples / sr
    if max_duration_s is not None and duration_s > max_duration_s:
        raise ValueError(
            f"Audio exceeds maximum allowed duration of {max_duration_s}s "
            f"(decoded {duration_s:.1f}s). Set VLLM_MAX_AUDIO_DECODE_DURATION_S to "
            "increase this limit."
        )
    if max_decode_bytes is not None:
        pcm_bytes = num_samples * num_channels * np.dtype(np.float32).itemsize
        if pcm_bytes > max_decode_bytes:
            raise ValueError(
                f"Audio would allocate {pcm_bytes / 2**20:.0f} MiB of PCM, exceeding the "
                f"{max_decode_bytes / 2**20:.0f} MiB limit. Set VLLM_MAX_AUDIO_DECODE_BYTES "
                "to increase this limit."
            )


_SEGMENT_MAX = int(os.environ.get("B10_DECODE_SEGMENTS", "1"))
_SEGMENT_TARGET_S = float(os.environ.get("B10_DECODE_SEGMENT_TARGET_S", "120"))
_SEGMENT_MARGIN_S = 0.25          # overlap compared between neighbours
_SEGMENT_WARMUP_S = 0.25          # extra lead-in discarded: codecs (mp3 bit reservoir) need a few frames after a seek
# One pool per API process bounds the extra threads segmenting adds: under heavy load
# segments queue here and a long file degrades to a serial decode, never to N x load.
_segment_pool = (
    ThreadPoolExecutor(int(os.environ.get("B10_DECODE_SEGMENT_WORKERS", "8")), thread_name_prefix="b10-segdecode")
    if _SEGMENT_MAX > 1 else None
)




def _to_mono(samples) -> np.ndarray:
    y = samples.numpy()
    if y.shape[0] == 1:
        return np.ascontiguousarray(y[0], dtype=np.float32)
    return np.mean(y, axis=0, dtype=np.float32)


def _decode_budget_s(sr: int, max_duration_s: float | None, max_decode_bytes: int | None) -> float | None:
    """Seconds of audio the caller's duration/byte limits allow; ``None`` when unlimited.

    Decoding stops one second past this budget. That bounds memory the way vLLM's frame loop
    does; it is not a cutoff: a file that reaches the extra second is longer than the budget
    and ``_check_bounds`` rejects it whole.
    """
    limits_s = []
    if max_duration_s is not None:
        limits_s.append(max_duration_s)
    if max_decode_bytes is not None:
        limits_s.append(max_decode_bytes / (sr * np.dtype(np.float32).itemsize))  # mono float32
    return min(limits_s) if limits_s else None


class _SegmentSpliceError(RuntimeError):
    """Adjacent segment decodes disagree on their shared overlap: splicing them is not safe."""


def _decode_torchcodec_segmented(
    data: bytes, sr: int, mono: bool, n: int, header_s: float, begin_s: float, stop_s: float | None
) -> np.ndarray:
    """Decode ``n`` time ranges in parallel and splice them on the sample grid.

    Sample k of the full decode plays at ``begin_s + k / sr`` (the stream's start offset, e.g.
    AAC priming), so every boundary is mapped through ``begin_s``. Each range is requested with a
    margin on both sides and trimmed to its exact sample span using the pts torchcodec returns.
    The margins double as a consistency check: neighbours must reproduce each other's samples
    bit for bit across the shared overlap, otherwise the codec/resampler state after a seek does
    not match a from-the-start decode and the caller falls back to a full decode. Files whose
    header lies about duration surface as decode errors here and take the same fallback.
    """
    from torchcodec.decoders import AudioDecoder

    total = int(round(header_s * sr))
    idx = [round(i * total / n) for i in range(n + 1)]
    margin = int(round(_SEGMENT_MARGIN_S * sr))

    def one(i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a, z = idx[i], idx[i + 1]
        start_s = max(begin_s + a / sr - _SEGMENT_MARGIN_S - _SEGMENT_WARMUP_S, 0.0)
        # The last range runs to the budget, not to EOF, so a header that understates the
        # duration cannot make it decode without bound.
        seg_stop_s = stop_s if i == n - 1 else begin_s + z / sr + _SEGMENT_MARGIN_S
        s = AudioDecoder(data, sample_rate=sr, num_channels=1 if mono else None)
        got = s.get_samples_played_in_range(start_s, seg_stop_s)
        first = int(round((begin_s + a / sr - float(got.pts_seconds)) * sr))
        if first < 0:
            raise _SegmentSpliceError(f"segment {i} started {-first} samples late")
        y = _to_mono(got.data)
        body = y[first:] if i == n - 1 else y[first : first + (z - a)]
        head = y[max(first - margin, 0) : first]            # samples before a, for the check with i-1
        tail = y[first + (z - a) : first + (z - a) + margin] # samples after z, for the check with i+1
        return head, body, tail

    parts = list(_segment_pool.map(one, range(n)))
    for i in range(n - 1):
        tail, next_head, next_body = parts[i][2], parts[i + 1][0], parts[i + 1][1]
        # tail of i must equal the start of i+1's body; head of i+1 must equal the end of i's body
        k = min(len(tail), len(next_body))
        j = min(len(next_head), len(parts[i][1]))
        if k == 0 or j == 0 or not np.array_equal(tail[:k], next_body[:k]) \
                or not np.array_equal(next_head[-j:], parts[i][1][-j:]):
            raise _SegmentSpliceError(f"segments {i} and {i + 1} disagree on their overlap")
    return np.concatenate([b for _, b, _ in parts])


def _decode_torchcodec(
    data: bytes, sr: int, mono: bool, max_duration_s: float | None, max_decode_bytes: int | None
) -> np.ndarray:
    from torchcodec.decoders import AudioDecoder

    decoder = AudioDecoder(data, sample_rate=sr, num_channels=1 if mono else None)
    # The header duration only sizes the parallel ranges; it is an estimate for some formats
    # (VBR MP3 without a length tag), so over-length files are judged on decoded samples
    # by _check_bounds, as vLLM's loader does, never on the header.
    header_s = decoder.metadata.duration_seconds_from_header
    begin_s = decoder.metadata.begin_stream_seconds_from_header
    budget_s = _decode_budget_s(sr, max_duration_s, max_decode_bytes)
    stop_s = None if budget_s is None else float(begin_s or 0.0) + budget_s + 1.0
    n = 1
    # Only a file that fits the budget is split: the ranges come from the header, so a header
    # longer than the budget would schedule decodes past it. Such a file takes the single
    # bounded decode below, which stops at the budget and rejects it.
    if _SEGMENT_MAX > 1 and header_s and begin_s is not None and (budget_s is None or header_s <= budget_s):
        n = max(1, min(_SEGMENT_MAX, int(header_s // _SEGMENT_TARGET_S)))
    if n > 1:
        try:
            y = _decode_torchcodec_segmented(data, sr, mono, n, header_s, float(begin_s), stop_s)
        except Exception as exc:  # noqa: BLE001 - any segment problem means: decode the normal way
            logger.warning("segmented decode fell back to a full decode (%s: %s)", type(exc).__name__, str(exc)[:120])
        else:
            _check_bounds(y.size, 1, sr, max_duration_s, max_decode_bytes)
            return y
    # (channels, num_samples), float32 in [-1, 1]; bounded by the budget, never by EOF alone
    samples = decoder.get_samples_played_in_range(0.0, stop_s).data
    _check_bounds(samples.shape[-1], samples.shape[0], sr, max_duration_s, max_decode_bytes)
    return _to_mono(samples)


def load_audio_b10(
    path: BytesIO | Path | str,
    *,
    sr: float | None = 22050,
    mono: bool = True,
    max_duration_s: float | None = None,
    max_decode_bytes: int | None = None,
) -> tuple[np.ndarray, int]:
    assert sr is not None, "the transcription path always passes the model sample rate"
    if not mono:
        raise ValueError("b10_decode returns mono only; the transcription path is mono")
    y = _decode_torchcodec(_read_bytes(path), int(sr), mono, max_duration_s, max_decode_bytes)
    return y, int(sr)
