"""Synthetic "speech" for the chunking tests: one square-wave burst per word.

Word k is a 0.8 s burst of amplitude ``level(k)`` followed by 0.2 s of silence (longer
where ``pause_after`` says). A fake engine reads the words back from a chunk's samples
(``read_words``), so a test can check that every word comes out of the stitched
transcript exactly once.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SR = 16_000
WORD_S = 0.8
GAP_S = 0.2


def level(k: int) -> float:
    return 0.1 + 0.004 * k


def word_audio(n_words: int, pause_after: dict[int, float] | None = None) -> tuple[np.ndarray, list[tuple[float, float]]]:
    pause_after = pause_after or {}
    pieces, words, t = [], [], 0.0
    square = np.where((np.arange(int(WORD_S * SR)) // 40) % 2 == 0, 1.0, -1.0).astype(np.float32)  # 200 Hz
    for k in range(n_words):
        pieces.append(square * level(k))
        words.append((t, t + WORD_S))
        t += WORD_S
        gap = pause_after.get(k, GAP_S)
        pieces.append(np.zeros(int(round(gap * SR)), dtype=np.float32))
        t += gap
    return np.concatenate(pieces), words


def read_words(samples: np.ndarray, min_run_s: float = 0.1) -> list[str]:
    """Word labels ("w<k>") of the bursts in ``samples`` (float, 16 kHz), in order."""
    loud = np.abs(samples) > 1e-3
    out, i, n = [], 0, len(samples)
    edges = np.flatnonzero(np.diff(loud.astype(np.int8))) + 1
    bounds = np.concatenate([[0], edges, [n]])
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if loud[lo] and hi - lo >= min_run_s * SR:
            amp = float(np.abs(samples[lo:hi]).mean())
            out.append(f"w{round((amp - 0.1) / 0.004)}")
    return out


def write_wav(path: Path, audio: np.ndarray, rate: int = SR, stereo: bool = False) -> None:
    if rate != SR:
        x = np.interp(np.arange(int(len(audio) * rate / SR)) * SR / rate, np.arange(len(audio)), audio)
    else:
        x = audio
    pcm = np.clip(np.rint(x * 32767), -32768, 32767).astype("<i2")
    if stereo:
        pcm = np.repeat(pcm[:, None], 2, axis=1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2 if stereo else 1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def write_m4a(path: Path, audio: np.ndarray) -> None:
    import av

    with av.open(str(path), "w", format="mp4") as c:
        s = c.add_stream("aac", rate=SR)
        s.layout = "mono"
        step = 1024
        for i in range(0, len(audio), step):
            f = av.AudioFrame.from_ndarray(audio[None, i : i + step].astype(np.float32), format="fltp", layout="mono")
            f.sample_rate = SR
            f.pts = i
            for p in s.encode(f):
                c.mux(p)
        for p in s.encode(None):
            c.mux(p)
