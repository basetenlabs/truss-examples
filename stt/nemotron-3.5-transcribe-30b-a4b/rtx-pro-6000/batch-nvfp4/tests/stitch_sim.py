"""Simulated chunk transcripts for the stitching tests: a reference transcript, the chunk
plan's spans, and each chunk "transcribed" with its own errors.

Words are spread evenly over the audio (TED talks run ~2.9 words/s). Each chunk gets the
reference words whose midpoint lies in its span, then:

- independent substitutions at ``sub_rate`` (so the overlap differs between the chunks),
- case and punctuation differences (sentence-case, commas, periods),
- edge damage: the later chunk's first word cut to a fragment or dropped, an extra word
  at the earlier chunk's end ("um", "Thank you."), or its last word missing.

``ideal`` is what a perfect stitch gives: each chunk's own words up to the midpoint of
the overlap. Its WER is the chunk-level WER; a stitcher should add (almost) nothing.
"""

from __future__ import annotations

import random
import re

_NORM = re.compile(r"[^\w']+")


def norm_words(text: str) -> list[str]:
    return [w for w in (_NORM.sub("", t.lower()).strip("'") for t in text.split()) if w]


def wer(hyp: str, ref: str, band: int = 200) -> tuple[int, int]:
    """(word edits, reference words), banded Levenshtein on normalized words."""
    h, r = norm_words(hyp), norm_words(ref)
    n, m = len(r), len(h)
    INF = 1 << 30
    prev = {j: j for j in range(min(m, band) + 1)}
    for i in range(1, n + 1):
        cur = {}
        lo, hi = max(0, i - band), min(m, i + band)
        for j in range(lo, hi + 1):
            if j == 0:
                cur[j] = i
                continue
            best = prev.get(j - 1, INF) + (r[i - 1] != h[j - 1])
            best = min(best, prev.get(j, INF) + 1, cur.get(j - 1, INF) + 1)
            cur[j] = best
        prev = cur
    return prev.get(m, INF), n


_FILLER_SUBS = ["the", "a", "and", "that", "it", "of", "to", "in", "is", "this"]


def synthetic_reference(n_words: int, seed: int = 0) -> list[str]:
    """Zipf-ish English-like words: frequent function words, repeated bigrams ("of the")."""
    rng = random.Random(seed)
    common = ["the", "and", "of", "to", "a", "in", "that", "is", "it", "you", "we", "this", "i", "so", "what"]
    content = [f"word{k}" for k in range(1500)]
    weights = [1 / (k + 5) for k in range(len(content))]  # Zipf-like: word0 ~3% of content
    out = []
    while len(out) < n_words:
        r = rng.random()
        if r < 0.15:
            out += ["of", "the"]
        elif r < 0.5:
            out.append(rng.choice(common))
        else:
            out.append(rng.choices(content, weights)[0])
    return out[:n_words]


def _decorate(words: list[str], rng: random.Random) -> list[str]:
    out, cap = [], True
    for w in words:
        t = w.capitalize() if cap or rng.random() < 0.02 else w
        cap = False
        r = rng.random()
        if r < 0.06:
            t += "."
            cap = True
        elif r < 0.14:
            t += ","
        out.append(t)
    return out


def chunk_transcripts(
    ref: list[str], spans_s: list[tuple[float, float]], *, words_per_s: float, sub_rate: float = 0.03,
    edge_damage: bool = True, seed: int = 0,
) -> tuple[list[str], str]:
    """(chunk texts, ideal stitched text) for ``ref`` cut at ``spans_s``."""
    rng = random.Random(seed)
    mids = [(k + 0.5) / words_per_s for k in range(len(ref))]
    texts, ideal_parts = [], []
    for ci, (s, e) in enumerate(spans_s):
        idx = [k for k, t in enumerate(mids) if s <= t < e]
        words = [ref[k] if rng.random() >= sub_rate else rng.choice(_FILLER_SUBS) for k in idx]
        own_lo = s if ci == 0 else (s + spans_s[ci - 1][1]) / 2
        own_hi = e if ci + 1 == len(spans_s) else (e + spans_s[ci + 1][0]) / 2
        ideal_parts += [w for k, w in zip(idx, words) if own_lo <= mids[k] < own_hi]
        dec = _decorate(words, rng)
        if edge_damage and dec:
            if ci > 0:
                r = rng.random()
                if r < 0.4:  # began mid-word
                    w0 = dec[0]
                    dec[0] = w0[len(w0) // 2 :] or w0
                elif r < 0.6:
                    dec = dec[1:]
            if ci + 1 < len(spans_s):
                r = rng.random()
                if r < 0.25:
                    dec.append(rng.choice(["um", "Thank you."]))
                elif r < 0.45:
                    dec = dec[:-1]
        texts.append(" ".join(dec))
    return texts, " ".join(ideal_parts)
