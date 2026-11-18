"""Cross-chunk speaker linking (``VV_LINK=embed``).

Each chunk is transcribed on its own, so VibeVoice numbers its speakers from 0 in every
chunk. To give one speaker one label across the file:

1. ``speaker_ranges``: for each (chunk, local speaker), pick up to ``max_s`` seconds of
   that speaker's own segments (longest first, edges trimmed, time where another
   local speaker also talks removed).
2. The proxy embeds that audio with WeSpeaker ResNet34-LM (``embed.py``) as soon as the
   chunk's reply arrives, overlapping with the other chunks' decoding.
3. ``cluster``: average-linkage agglomerative clustering on cosine similarity with
   cannot-link between speakers of the same chunk, stopping below ``threshold``.
   Items with less than ``min_s`` of usable speech (unreliable embeddings, usually a
   few words) join their most similar compatible cluster instead of seeding one. An item
   too short to embed at all (< 0.2 s, typically a word cut at a chunk boundary) takes
   the label of the speaker talking across that boundary in the neighbouring chunk.
4. Clusters are renumbered by first appearance, so a linked file looks like a
   single-pass reply (speakers "0", "1", ...).

Pure numpy; no torch import here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .transcript import Segment

TRIM_S = 0.2  # trimmed from both ends of a segment (timestamps are coarse)
MIN_PIECE_S = 0.3  # shorter usable pieces are dropped


@dataclass
class Item:
    chunk: int
    speaker: str
    first_start: float  # absolute start of the speaker's first segment in the chunk
    speech_s: float = 0.0  # seconds of audio that went into the embedding
    emb: np.ndarray | None = None
    start_gap: float = float("inf")  # chunk start -> this speaker's first segment (s)
    end_gap: float = float("inf")  # this speaker's last segment end -> chunk end (s)


BOUNDARY_S = 1.0  # a segment this close to a cut is treated as spanning it


def make_items(chunk: int, offset_s: float, length: float, segs: list[Segment],
               ranges: dict[str, list[tuple[float, float]]]) -> list[Item]:
    """One ``Item`` (without embedding) per speaker in ``ranges``."""
    out = []
    for spk, rg in ranges.items():
        own = [sg for sg in segs if sg.speaker == spk]
        first = max(0.0, min(sg.start for sg in own))
        last = min(length, max(sg.end for sg in own))
        out.append(Item(chunk, spk, offset_s + first, sum(b - a for a, b in rg), None,
                        first, max(0.0, length - last)))
    return out


def _subtract(a: float, b: float, others: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out = [(a, b)]
    for oa, ob in others:
        nxt = []
        for x, y in out:
            if ob <= x or oa >= y:
                nxt.append((x, y))
                continue
            if oa > x:
                nxt.append((x, oa))
            if ob < y:
                nxt.append((ob, y))
        out = nxt
    return out


def speaker_ranges(segs: list[Segment], length: float, max_s: float) -> dict[str, list[tuple[float, float]]]:
    """Chunk-relative ``[(start_s, end_s), ...]`` per local speaker, at most ``max_s`` s each."""
    by_spk: dict[str, list[tuple[float, float]]] = {}
    for s in segs:
        if s.speaker == "unknown":
            continue
        a, b = min(max(s.start, 0.0), length), min(max(s.end, 0.0), length)
        if b > a:
            by_spk.setdefault(s.speaker, []).append((a, b))
    out: dict[str, list[tuple[float, float]]] = {}
    for spk, ivs in by_spk.items():
        others = [iv for k, v in by_spk.items() if k != spk for iv in v]
        pieces: list[tuple[float, float]] = []
        for a, b in ivs:
            if b - a > 2 * TRIM_S + MIN_PIECE_S:
                a, b = a + TRIM_S, b - TRIM_S
            pieces.extend(p for p in _subtract(a, b, others) if p[1] - p[0] >= MIN_PIECE_S)
        if not pieces:  # nothing clean: fall back to the raw segments
            pieces = [(a, b) for a, b in ivs]
        pieces.sort(key=lambda p: p[1] - p[0], reverse=True)
        chosen, total = [], 0.0
        for a, b in pieces:
            if total >= max_s:
                break
            b = min(b, a + (max_s - total))
            chosen.append((a, b))
            total += b - a
        out[spk] = sorted(chosen)
    return out


def gather(pcm: np.ndarray, rate: int, offset_samples: int, ranges: list[tuple[float, float]]) -> np.ndarray:
    """Concatenate ``ranges`` (chunk-relative seconds) of ``pcm`` starting at ``offset_samples``."""
    parts = [pcm[offset_samples + int(a * rate): offset_samples + int(b * rate)] for a, b in ranges]
    parts = [p for p in parts if p.size]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=pcm.dtype)


def _agglomerate(embs: np.ndarray, chunks: list[int], threshold: float) -> list[int]:
    """Average-linkage clustering with cannot-link inside a chunk; returns a cluster per row."""
    n = len(chunks)
    if n == 0:
        return []
    sim = embs @ embs.T
    w = sim.astype(np.float64).copy()  # w[i, j] = sum of pairwise sims between clusters i and j
    size = np.ones(n)
    members = [{i} for i in range(n)]
    cset = [{chunks[i]} for i in range(n)]
    alive = list(range(n))
    while len(alive) > 1:
        best, pair = -np.inf, None
        for x in range(len(alive)):
            i = alive[x]
            for j in alive[x + 1:]:
                if cset[i] & cset[j]:
                    continue
                s = w[i, j] / (size[i] * size[j])
                if s > best:
                    best, pair = s, (i, j)
        if pair is None or best < threshold:
            break
        i, j = pair
        w[i, :] += w[j, :]
        w[:, i] += w[:, j]
        size[i] += size[j]
        members[i] |= members[j]
        cset[i] |= cset[j]
        alive.remove(j)
    label = [0] * n
    for c, i in enumerate(alive):
        for m in members[i]:
            label[m] = c
    return label


def cluster(items: list[Item], threshold: float, min_s: float) -> list[int]:
    """A global cluster index per item (renumbered by first appearance)."""
    raw = [-1] * len(items)
    have = [k for k, it in enumerate(items) if it.emb is not None]
    seeds = [k for k in have if items[k].speech_s >= min_s] or have
    if seeds:
        e = np.stack([items[k].emb for k in seeds]).astype(np.float32)
        e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
        lab = _agglomerate(e, [items[k].chunk for k in seeds], threshold)
        for k, c in zip(seeds, lab):
            raw[k] = c
    n_clusters = max(raw) + 1 if seeds else 0
    # Short items: join the most similar cluster that has no speaker from the same chunk.
    for k in have:
        if raw[k] != -1:
            continue
        best, arg = -np.inf, -1
        for c in range(n_clusters):
            mem = [m for m in range(len(items)) if raw[m] == c]
            if any(items[m].chunk == items[k].chunk for m in mem):
                continue
            s = float(np.mean([items[k].emb @ items[m].emb for m in mem]))
            if s > best:
                best, arg = s, c
        if arg == -1:
            arg, n_clusters = n_clusters, n_clusters + 1
        raw[k] = arg
    def compatible(k: int, c: int) -> bool:
        return all(not (raw[m] == c and items[m].chunk == items[k].chunk) for m in range(len(items)) if m != k)

    # No embedding: follow the speaker talking across the adjacent cut, else a new label.
    for k, it in enumerate(items):
        if raw[k] != -1:
            continue
        cands = []
        if it.start_gap <= BOUNDARY_S:
            prev = [m for m, jt in enumerate(items) if jt.chunk == it.chunk - 1 and raw[m] != -1]
            cands += sorted(((items[m].end_gap, m) for m in prev if items[m].end_gap <= BOUNDARY_S))[:1]
        if it.end_gap <= BOUNDARY_S:
            nxt = [m for m, jt in enumerate(items) if jt.chunk == it.chunk + 1 and raw[m] != -1]
            cands += sorted(((items[m].start_gap, m) for m in nxt if items[m].start_gap <= BOUNDARY_S))[:1]
        for _, m in sorted(cands):
            if compatible(k, raw[m]):
                raw[k] = raw[m]
                break
        else:
            raw[k], n_clusters = n_clusters, n_clusters + 1
    order = sorted(range(len(items)), key=lambda k: (items[k].first_start, items[k].chunk))
    remap: dict[int, int] = {}
    for k in order:
        remap.setdefault(raw[k], len(remap))
    return [remap[r] for r in raw]


def labels(items: list[Item], threshold: float, min_s: float) -> dict[tuple[int, str], str]:
    """``{(chunk, local_speaker): global_label}`` for ``transcript.stitch``."""
    return {(it.chunk, it.speaker): str(c) for it, c in zip(items, cluster(items, threshold, min_s))}
