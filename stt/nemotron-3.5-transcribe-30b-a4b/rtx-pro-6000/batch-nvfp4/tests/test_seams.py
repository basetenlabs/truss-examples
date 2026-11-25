"""Seam quality: stitching simulated chunk transcripts, the chunk planner on realistic
durations, and the exact audio each chunk carries. Run as tests/test_long_audio.py says.

Stitching is measured the way the benchmark measures it: WER of the stitched text against
the reference, next to the WER of an ideal stitch of the same chunk transcripts (each
chunk's own words up to the middle of the overlap). The difference is what the seams add.
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import long_audio as la  # noqa: E402
import stitch_sim as sim  # noqa: E402

SR = la.SAMPLE_RATE
CFG_600 = la.ChunkConfig(max_seconds=600, trigger_seconds=600, overlap_seconds=5, search_seconds=30)
CFG_DEFAULT = la.ChunkConfig()


# --- stitching -----------------------------------------------------------------------------


def _seam_cost(sub_rate: float, talks: int, words: int = 4000, words_per_s: float = 2.9) -> tuple[float, float, list]:
    edits = ideal_edits = ref_words = 0
    seams: list[la.Seam] = []
    for seed in range(talks):
        ref = sim.synthetic_reference(words, seed)
        n = int(words / words_per_s * SR)
        rng = np.random.default_rng(seed)
        energy = rng.uniform(0.5, 1.0, n // int(la.FRAME_S * SR)).astype(np.float32)
        spans = la.plan_chunks(energy, n, CFG_600)
        texts, ideal = sim.chunk_transcripts(
            ref, [(s / SR, e / SR) for s, e in spans], words_per_s=words_per_s, sub_rate=sub_rate, seed=seed
        )
        out, sm = la.stitch_texts(texts, CFG_600.overlap_seconds)
        seams += sm
        e, r = sim.wer(out, " ".join(ref))
        e0, _ = sim.wer(ideal, " ".join(ref))
        edits, ideal_edits, ref_words = edits + e, ideal_edits + e0, ref_words + r
    return 100 * edits / ref_words, 100 * ideal_edits / ref_words, seams


@pytest.mark.parametrize("sub_rate", [0.03, 0.08, 0.15])
def test_seams_add_at_most_a_tenth_of_a_point(sub_rate):
    """~23-minute talks in 600 s chunks; the overlap transcribed differently by each chunk
    (substitutions, case, punctuation, a fragment or missing word at the later chunk's
    start, an extra or missing word at the earlier chunk's end)."""
    stitched, ideal, seams = _seam_cost(sub_rate, talks=12)
    assert stitched - ideal <= 0.1, (stitched, ideal)
    assert all(s.matched >= 2 for s in seams)
    # Each seam drops about one overlap's worth of words in total (5 s, ~15 words), never a span.
    assert all(s.dropped_prev + s.dropped_next <= 30 for s in seams)


def test_repeated_phrase_outside_the_overlap_does_not_anchor_the_seam():
    # "of the" occurs early in prev's window and late in next's window; the true overlap
    # is "five six seven eight nine ten" with one substitution and a fragment.
    prev = "one of the two three four five six seven eight nine ten"
    nxt = "ive six seven ate nine ten eleven twelve thirteen fourteen of the fifteen"
    text, seam = la.stitch_pair(prev, nxt, 5)
    assert text == "one of the two three four five six seven eight nine ten eleven twelve thirteen fourteen of the fifteen"
    assert seam.matched == 4


def test_case_and_punctuation_do_not_break_the_alignment():
    prev = "and so we built it. It worked, Mostly"
    nxt = "it Worked. mostly, and then we scaled"
    text, seam = la.stitch_pair(prev, nxt, 5)
    assert sim.norm_words(text) == sim.norm_words("and so we built it it worked mostly and then we scaled")
    assert seam.matched == 3


def test_join_point_is_the_middle_of_the_overlap():
    # Earlier chunk garbles its last overlap word, later chunk garbles its first: the join
    # keeps the earlier chunk's early overlap words and the later chunk's late ones.
    prev = "alpha beta gamma delta epsilon zeta eta THETA-BAD"
    nxt = "BAD-GAMMA delta epsilon zeta eta theta iota kappa"
    text, seam = la.stitch_pair(prev, nxt, 5)
    assert text == "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    assert seam.matched == 4


def test_extra_and_missing_words_near_the_edge():
    prev = "we went to the market and bought apples um"
    nxt = "ket and apples and pears for the week"  # fragment, "bought" missing
    text, _ = la.stitch_pair(prev, nxt, 5)
    assert text == "we went to the market and bought apples and pears for the week"


def test_unrelated_chunk_text_is_concatenated_not_spliced():
    prev = "the energy problem needs three things to be solved by twenty fifty"
    nxt = "Here is a summary of the talk you shared with me"
    text, seam = la.stitch_pair(prev, nxt, 5)
    assert seam == la.Seam(0, 0, 0) and text == prev + " " + nxt


# --- planning ------------------------------------------------------------------------------


def _check_plan(spans, n, cfg):
    limit, overlap = int(cfg.max_seconds * SR), int(cfg.overlap_seconds * SR)
    assert spans[0][0] == 0 and spans[-1][1] == n
    for (s0, e0), (s1, _) in zip(spans, spans[1:]):
        assert e0 - s1 == overlap
    lengths = [e - s for s, e in spans]
    assert max(lengths[:-1], default=0) <= limit
    # A small tail folded into the last chunk may take it past max_seconds, never past
    # the longest single request.
    assert lengths[-1] <= max(limit, int(cfg.trigger_seconds * SR))
    if len(spans) > 1:
        assert min(lengths) >= la.MIN_TAIL_FRACTION * limit, [x / SR for x in lengths]
    return lengths


@pytest.mark.parametrize("seconds", list(range(601, 1301, 37)) + [1249.02, 1201, 1300])
def test_600_s_chunks_have_no_tiny_tail(seconds):
    n = int(seconds * SR)
    energy = np.random.default_rng(int(seconds)).uniform(0.5, 1, n // 320).astype(np.float32)
    _check_plan(la.plan_chunks(energy, n, CFG_600), n, CFG_600)


@pytest.mark.parametrize("seconds", [3601, 3650, 3700, 3960, 4000, 5000, 6500, 6600, 6700, 7200])
def test_default_chunks_have_no_tiny_tail(seconds):
    n = int(seconds * SR)
    energy = np.random.default_rng(seconds).uniform(0.5, 1, n // 320).astype(np.float32)
    lengths = _check_plan(la.plan_chunks(energy, n, CFG_DEFAULT), n, CFG_DEFAULT)
    # As many chunks as the greedy plan, one fewer when a small tail was folded in.
    assert len(lengths) in (int(np.ceil((seconds - 5) / 3265)), int(np.ceil((seconds - 5) / 3265)) - 1)


def test_small_tail_merges_when_the_result_fits_one_request():
    cfg = la.ChunkConfig(max_seconds=600, trigger_seconds=900, overlap_seconds=5, search_seconds=30)
    n = 650 * SR  # greedy: ~600 s + ~55 s
    spans = la.plan_chunks(np.ones(n // 320, dtype=np.float32), n, cfg)
    assert spans == [(0, n)] or len(spans) == 1  # 650 s <= trigger: one request anyway
    n = 1250 * SR  # greedy: 600 + 600 + ~60; merging gives ~655 s <= 900
    spans = la.plan_chunks(np.ones(n // 320, dtype=np.float32), n, cfg)
    assert len(spans) == 2 and spans[-1][1] - spans[-1][0] <= 900 * SR


def test_rebalance_keeps_cuts_in_pauses():
    # 1249 s at 600 s: greedy leaves ~79 s, so three ~420 s chunks. Pauses near 420 s and
    # 838 s are inside the even plan's search windows and are where it cuts.
    n = int(1249.02 * SR)
    e = np.ones(n // 320, dtype=np.float32)
    for t in (418.0, 836.0):
        e[int(t / la.FRAME_S) : int((t + 0.5) / la.FRAME_S)] = 1e-6
    spans = la.plan_chunks(e, n, CFG_600)
    assert len(spans) == 3
    assert 418 * SR <= spans[0][1] <= 418.5 * SR
    assert 836 * SR <= spans[1][1] <= 836.5 * SR


def test_cut_search_window_is_the_last_search_seconds_before_the_limit():
    n = 1000 * SR  # greedy: 600 + 405, no rebalancing
    e = np.ones(n // 320, dtype=np.float32)
    e[int(560 / la.FRAME_S) : int(561 / la.FRAME_S)] = 1e-6  # 40 s before the limit: outside
    e[int(575 / la.FRAME_S) : int(575.5 / la.FRAME_S)] = 1e-6  # 25 s before: inside
    spans = la.plan_chunks(e, n, CFG_600)
    assert 575 * SR <= spans[0][1] <= 575.5 * SR
    e[int(575 / la.FRAME_S) : int(575.5 / la.FRAME_S)] = 1  # no pause in the window
    spans = la.plan_chunks(e, n, CFG_600)
    assert 570 * SR <= spans[0][1] <= 600 * SR


# --- the audio each chunk carries ------------------------------------------------------------


def test_each_chunk_wav_is_exactly_its_span_of_the_source(tmp_path):
    rng = np.random.default_rng(7)
    seconds = 1249
    pcm = rng.integers(-20000, 20000, seconds * SR, dtype=np.int16)
    # Quieter stretches so the planner has pauses to find.
    for t in range(0, seconds, 9):
        pcm[t * SR : t * SR + SR // 4] //= 200
    src = tmp_path / "talk.wav"
    with wave.open(str(src), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.astype("<i2").tobytes())
    out = tmp_path / "chunks"
    out.mkdir()
    cfg = la.ChunkConfig(max_seconds=600, trigger_seconds=600, overlap_seconds=5, search_seconds=30, max_audio_seconds=3600)
    chunks = la.decode_and_split(str(src), str(out), cfg)
    assert len(chunks) == 3
    for ch in chunks:
        with wave.open(ch.path) as w:
            got = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        assert np.array_equal(got, pcm[ch.start : ch.end]), ch
