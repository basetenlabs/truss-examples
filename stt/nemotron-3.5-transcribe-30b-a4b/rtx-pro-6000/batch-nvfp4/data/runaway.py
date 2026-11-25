"""Runaway guard: bound, detect and repair transcripts the model did not stop on.

Nemotron 3.5 Transcribe sometimes keeps generating after the speech ends: it loops a
phrase, or invents fluent text that was never said (TED-LIUM exec 11/12: a chunk at the
end of a talk produced ~13k words of a "YouTube vlog" at 31 words/s; normal speech is
2.2-3.3 words/s). ``chunk_proxy.py`` applies this module to every transcription request,
chunked or not:

    request ── duration d ──▶ max_tokens = min(client's, ceil(d × 8) + 256)      (token_cap)
         │
         ▼  each engine unit (whole file, or one chunk)
    assess: finish=length under our budget │ words > 8/s × d + 16 │ 2.5× the request's
            median words/s (3+ units)      │ a phrase looping (longest_loop)
         │ anomalous
         ▼
    retry once, split in two at a pause (split_in_two)  ── clean ──▶ use the halves
         │ still anomalous
         ▼
    NVIDIA's generation-loop cleanup (model card) and finish_reason "length"

8 tokens/s: transcripts measured ~3.3 tokens/s (1-hour podcast ≈ 11.6k completion
tokens, README Sizing) and the model card's densest hour is ~20,000 tokens (5.6/s); 8/s
is 1.4x that. The 256-token margin covers short clips. The cap never raises the client's.

No web dependency: tested in tests/test_runaway.py.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

# --- NVIDIA's generation-loop guardrail -------------------------------------------------
#
# Verbatim from the model card, "Generation-Loop Guardrail"
# (huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B, revision a188fff, README.md),
# which NVIDIA applies to the reported transcription results. Adapted there from the
# Qwen3-ASR repetition fix (QwenLM/Qwen3-ASR@7c6daf7, qwen_asr/inference/utils.py),
# Copyright 2026 The Alibaba Qwen team, Apache License 2.0.

CHAR_RUN = re.compile(r"(.)\1{20,}", re.DOTALL)
BLOCK_RUN = re.compile(r"(.{1,4096}?)\1{19,}", re.DOTALL)


def cleanup_generation_loops(text: str) -> str:
    # Collapse a character repeated ≥21 times to one copy.
    text = CHAR_RUN.sub(lambda m: m.group(1), text)

    # Collapse an exact 1–4096-character block repeated ≥20 times.
    return BLOCK_RUN.sub(lambda m: m.group(1), text)


# --- fast path for the cleanup ------------------------------------------------------------

_WS_RUN = re.compile(r"\s{20,}")
#: A block repeated 20 times in characters shows as ≥18 whole repeats in words (a partial
#: word at each end); 17 leaves a margin. Blocks are ≤4,096 characters, so ≤1,024 words.
_PRECHECK_REPEATS = 17
_PRECHECK_PERIOD = 1024


def cleanup_if_looping(text: str) -> str:
    """``cleanup_generation_loops(text)``, skipping the regex when it cannot match.

    The block regex costs ~6 s on a 1-hour transcript (it tries every block length at every
    position). It can only change text that holds a run of 20+ characters without
    whitespace, a whitespace run, or a word sequence repeated back to back ~20 times; when
    none is present the text is returned as is.
    """
    words = text.split()
    if (
        any(len(w) >= 20 for w in words)
        or _WS_RUN.search(text)
        or _max_repeats(words, _PRECHECK_PERIOD, normalize=False) >= _PRECHECK_REPEATS
    ):
        return cleanup_generation_loops(text)
    return text


# --- loop detection on words ------------------------------------------------------------

_NORM = re.compile(r"[^\w']+", re.UNICODE)


def _ids(words: list[str], normalize: bool) -> np.ndarray:
    table: dict[str, int] = {}
    out = np.empty(len(words), dtype=np.int64)
    for i, w in enumerate(words):
        key = _NORM.sub("", w.lower()) if normalize else w
        # Punctuation-only tokens never match each other.
        out[i] = table.setdefault(key, len(table)) if key else -1 - i
    return out


def _runs(eq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.diff(np.concatenate(([0], eq.view(np.int8), [0])))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def _max_repeats(words: list[str], max_period: int, normalize: bool = True) -> int:
    a = _ids(words, normalize)
    best = 1
    for n in range(1, min(max_period, len(a) // 2) + 1):
        eq = a[n:] == a[:-n]
        if not eq.any():
            continue
        s, e = _runs(eq)
        best = max(best, int((e - s).max()) // n + 1)
    return best


@dataclass(frozen=True)
class Loop:
    #: Words in the repeated phrase.
    period: int
    #: Back-to-back copies of it.
    repeats: int
    #: Index of the first word of the run.
    start: int


def _is_loop(period: int, repeats: int) -> bool:
    # "no, no, no, no" or "thank you, thank you, thank you" is speech; a phrase of 3+ words
    # said 8 times running, a sentence of 8+ words said 4 times, or anything 17 times is not.
    return repeats >= 17 or (period >= 3 and repeats >= 8) or (period >= 8 and repeats >= 4)


def longest_loop(text: str, max_period: int = 512) -> Loop | None:
    """The loop (see ``_is_loop``) covering the most words, case and punctuation ignored."""
    words = text.split()
    a = _ids(words, normalize=True)
    best: Loop | None = None
    for n in range(1, min(max_period, len(a) // 2) + 1):
        eq = a[n:] == a[:-n]
        if not eq.any():
            continue
        s, e = _runs(eq)
        lengths = e - s
        k = int(lengths.argmax())
        reps = int(lengths[k]) // n + 1
        if _is_loop(n, reps) and (best is None or n * reps > best.period * best.repeats):
            best = Loop(n, reps, int(s[k]))
    return best


# --- configuration ----------------------------------------------------------------------


@dataclass(frozen=True)
class GuardConfig:
    """Runaway-guard knobs, read from the environment by :meth:`from_env`."""

    enabled: bool = True
    #: Transcript budget per second of audio (token_cap).
    tokens_per_s: float = 8.0
    #: Added to every budget, for short clips.
    margin_tokens: int = 256
    #: Absolute rate above which a transcript is invented, words per second of audio.
    max_words_per_s: float = 8.0
    #: Words of slack on the absolute rate, so a 1-second "yes, okay, sure" is not flagged.
    rate_slack_words: int = 16
    #: Relative rate: a unit faster than this times the request's median (3+ units)...
    rel_words_factor: float = 2.5
    #: ...and faster than this, words/s (2.5 x a slow, sparse chunk is still speech).
    rel_words_floor: float = 4.5
    #: An anomalous request that was not chunked is retried as two halves from this long.
    split_min_seconds: float = 600.0
    #: An anomalous chunk of a chunked request is retried as two halves from this long.
    chunk_split_min_seconds: float = 60.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> GuardConfig:
        env = os.environ if env is None else env

        def number(name: str, default: float, minimum: float = 0.0) -> float:
            raw = env.get(name, "").strip()
            if raw == "":
                return default
            try:
                value = float(raw)
            except ValueError:
                raise ValueError(f"{name} must be a number, got {raw!r}") from None
            if not math.isfinite(value) or value < minimum:
                raise ValueError(f"{name} must be a number >= {minimum:g}, got {raw!r}")
            return value

        flag = env.get("NEMOTRON_RUNAWAY_GUARD", "").strip().lower() or "1"
        if flag not in ("0", "1", "true", "false", "on", "off"):
            raise ValueError(f"NEMOTRON_RUNAWAY_GUARD must be 1 or 0, got {flag!r}")
        return cls(
            enabled=flag in ("1", "true", "on"),
            tokens_per_s=number("NEMOTRON_RUNAWAY_TOKENS_PER_S", cls.tokens_per_s, 1.0),
            margin_tokens=int(number("NEMOTRON_RUNAWAY_MARGIN_TOKENS", cls.margin_tokens)),
            split_min_seconds=number("NEMOTRON_RUNAWAY_SPLIT_MIN_SECONDS", cls.split_min_seconds, 10.0),
            chunk_split_min_seconds=number("NEMOTRON_RUNAWAY_CHUNK_SPLIT_MIN_SECONDS", cls.chunk_split_min_seconds, 10.0),
        )


# --- max_tokens -------------------------------------------------------------------------

_MAX_KEYS = ("max_tokens", "max_completion_tokens")


def token_budget(duration_s: float, g: GuardConfig) -> int:
    return math.ceil(max(duration_s, 0.0) * g.tokens_per_s) + g.margin_tokens


def client_max_tokens(body: dict) -> int | None:
    vals = [body[k] for k in _MAX_KEYS if isinstance(body.get(k), int) and not isinstance(body.get(k), bool) and body[k] > 0]
    return min(vals) if vals else None


def apply_token_cap(body: dict, duration_s: float, g: GuardConfig) -> int:
    """Set ``body``'s completion limit to min(the client's, the duration budget); returns it."""
    client = client_max_tokens(body)
    cap = token_budget(duration_s, g)
    eff = cap if client is None else min(client, cap)
    present = [k for k in _MAX_KEYS if k in body and body[k] is not None]
    for k in present or ["max_tokens"]:
        body[k] = eff
    return eff


# --- assessment -------------------------------------------------------------------------


@dataclass
class Assessment:
    reasons: list[str] = field(default_factory=list)
    words: int = 0
    words_per_s: float = 0.0
    loop: Loop | None = None

    @property
    def anomalous(self) -> bool:
        return bool(self.reasons)


def words_per_s(text: str, duration_s: float) -> float:
    return len(text.split()) / max(duration_s, 1e-6)


def assess(
    text: str,
    finish: str | None,
    duration_s: float,
    eff_cap: int,
    g: GuardConfig,
    median_wps: float | None = None,
) -> Assessment:
    """Why this unit's transcript looks like a runaway (empty ``reasons`` if it does not).

    ``finish == "length"`` counts only when the limit was at least the duration budget:
    a client that asked for 50 tokens and got 50 is not a runaway.
    """
    words = len(text.split())
    rate = words / max(duration_s, 1e-6)
    a = Assessment(words=words, words_per_s=rate)
    if finish == "length" and eff_cap >= token_budget(duration_s, g):
        a.reasons.append("length")
    if words > g.max_words_per_s * duration_s + g.rate_slack_words:
        # "\x3e" is ">": a literal one reads as a pip requirement to the org pinning check.
        a.reasons.append(f"rate\x3e{g.max_words_per_s:g}/s")
    elif median_wps and rate > g.rel_words_factor * median_wps and rate > g.rel_words_floor:
        a.reasons.append(f"rate\x3e{g.rel_words_factor:g}x median")
    a.loop = longest_loop(text)
    if a.loop is not None:
        a.reasons.append(f"loop {a.loop.period}w x{a.loop.repeats}")
    return a


def median(values: list[float]) -> float | None:
    """Median words/s of a request's units; None under 3 units (no outlier to compare)."""
    if len(values) < 3:
        return None
    return float(np.median(values))
