"""Chunk boundaries for long audio (lever 1).

The cut near each ``chunk_s`` mark goes to the quietest 0.5 s frame within
``+/- search_s`` (energy minimum). No overlap: every sample lands in exactly one
chunk. VAD-based cuts and speaker linking are later study arms.
"""

from __future__ import annotations

import numpy as np

FRAME_S = 0.5


def bounds(pcm: np.ndarray, sample_rate: int, chunk_s: float, search_s: float) -> list[tuple[int, int]]:
    """``[(start_sample, end_sample), ...]`` covering ``pcm`` with no gaps or overlap."""
    n = int(pcm.shape[0])
    step = int(chunk_s * sample_rate)
    if chunk_s <= 0 or n <= step:
        return [(0, n)]
    frame = max(1, int(FRAME_S * sample_rate))
    search = int(max(0.0, search_s) * sample_rate)
    cuts = [0]
    while n - cuts[-1] > step:
        target = cuts[-1] + step
        lo = max(cuts[-1] + frame, target - search)
        hi = min(n - frame, target + search)
        cut = target
        if hi > lo:
            starts = np.arange(lo, hi - frame + 1, frame // 2 or 1)
            if starts.size:
                # float32 of the search window only (not the whole file); same values as before.
                x = pcm[lo : int(starts[-1]) + frame].astype(np.float32)
                energy = np.array([float(np.mean(x[s - lo : s - lo + frame] ** 2)) for s in starts])
                cut = int(starts[int(np.argmin(energy))] + frame // 2)
        cut = min(max(cut, cuts[-1] + 1), n)
        cuts.append(cut)
    # Fold a tiny tail (< 10% of a chunk) into the last chunk.
    if len(cuts) > 1 and n - cuts[-1] < step * 0.1:
        cuts.pop()
    cuts.append(n)
    return list(zip(cuts[:-1], cuts[1:]))
