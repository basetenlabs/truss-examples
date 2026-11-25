"""Planning, decode and stitching in data/long_audio.py, on synthetic audio.

Not run by this repository's CI (no Python test job for presets). Run by hand:

    uv run --with pytest --with pytest-asyncio --with numpy --with aiohttp --with av==19.0.2 \
        pytest stt/nemotron-3.5-transcribe-30b-a4b/rtx-pro-6000/batch-nvfp4/tests
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
import long_audio as la  # noqa: E402
from synth import word_audio, write_m4a, write_wav  # noqa: E402

SR = la.SAMPLE_RATE
ENV = {"VLLM_MAX_AUDIO_DECODE_DURATION_S": "3600", "VLLM_MAX_AUDIO_CLIP_FILESIZE_MB": "180"}


# --- configuration ------------------------------------------------------------------------


def test_defaults_are_nvidia_long_form_setup():
    cfg = la.ChunkConfig.from_env(ENV)
    assert (cfg.trigger_seconds, cfg.max_seconds, cfg.overlap_seconds, cfg.enabled) == (3600, 3300, 5, True)
    assert cfg.max_audio_seconds >= 2 * 3600
    assert cfg.engine_max_file_bytes == 180 * 2**20


def test_zero_disables_without_other_checks():
    cfg = la.ChunkConfig.from_env({"NEMOTRON_CHUNK_MAX_SECONDS": "0"})
    assert not cfg.enabled


@pytest.mark.parametrize(
    "env, message",
    [
        ({"NEMOTRON_CHUNK_TRIGGER_SECONDS": "4000"}, "decode cap"),
        ({"NEMOTRON_CHUNK_TRIGGER_SECONDS": "3000"}, "below"),
        ({"NEMOTRON_CHUNK_MAX_SECONDS": "abc"}, "number"),
        ({"NEMOTRON_CHUNK_OVERLAP_SECONDS": "-1"}, "non-negative"),
        ({"NEMOTRON_CHUNK_MAX_SECONDS": "60", "NEMOTRON_CHUNK_TRIGGER_SECONDS": "60"}, "half"),
        ({"NEMOTRON_CHUNK_MAX_SECONDS": "10", "NEMOTRON_CHUNK_TRIGGER_SECONDS": "10"}, "at least 30"),
        ({"NEMOTRON_MAX_AUDIO_SECONDS": "1000"}, "below"),
        ({"VLLM_MAX_AUDIO_CLIP_FILESIZE_MB": "25"}, "VLLM_MAX_AUDIO_CLIP_FILESIZE_MB"),
    ],
)
def test_misconfiguration_fails_loudly(env, message):
    with pytest.raises(ValueError, match=message):
        la.ChunkConfig.from_env({**ENV, **env})


# --- planning -----------------------------------------------------------------------------


def _energy(n_samples: int, quiet: list[tuple[float, float]]) -> np.ndarray:
    frames = n_samples // int(la.FRAME_S * SR)
    e = np.ones(frames, dtype=np.float32)
    for lo, hi in quiet:
        e[int(lo / la.FRAME_S) : int(hi / la.FRAME_S)] = 1e-6
    return e


def test_plan_cuts_in_pause_within_limit():
    cfg = la.ChunkConfig(max_seconds=3300, trigger_seconds=3300, overlap_seconds=5, search_seconds=30)
    n = 9000 * SR  # a 2,445 s last chunk: no tail rebalancing
    # Pauses at 3290-3291 s (in chunk 0's search window) and 6560-6561 s (chunk 1's).
    spans = la.plan_chunks(_energy(n, [(3290.0, 3291.0), (6560.0, 6561.0)]), n, cfg)
    assert len(spans) == 3
    assert spans[0][0] == 0 and spans[-1][1] == n
    for (s, e) in spans:
        assert e - s <= 3300 * SR
    assert 3290 * SR <= spans[0][1] <= 3291 * SR
    # Each chunk starts `overlap` before the previous cut.
    for (_, cut), (start, _) in zip(spans, spans[1:]):
        assert cut - start == 5 * SR
    assert 6560 * SR <= spans[1][1] <= 6561 * SR


def test_plan_without_pause_still_bounded():
    cfg = la.ChunkConfig(max_seconds=3300, trigger_seconds=3300, overlap_seconds=5, search_seconds=30)
    n = int(7000.5 * SR)
    spans = la.plan_chunks(np.ones(n // 320, dtype=np.float32), n, cfg)
    assert all(e - s <= 3300 * SR for s, e in spans)
    assert spans[-1][1] == n


@pytest.mark.parametrize("seconds", [10.0, 3300.0])
def test_short_audio_is_one_span(seconds):
    cfg = la.ChunkConfig()
    n = int(seconds * SR)
    assert la.plan_chunks(np.ones(n // 320, dtype=np.float32), n, cfg) == [(0, n)]


def test_one_hour_is_not_chunked_but_61_minutes_is():
    """Up to the 1-hour trigger: one engine request. Past it: 55-min chunks."""
    cfg = la.ChunkConfig()
    n = 3600 * SR
    assert la.plan_chunks(np.ones(n // 320, dtype=np.float32), n, cfg) == [(0, n)]
    n = 3660 * SR
    spans = la.plan_chunks(np.ones(n // 320, dtype=np.float32), n, cfg)
    assert len(spans) == 2 and all(e - s <= 3300 * SR for s, e in spans)


def test_forced_variant_both_120():
    cfg = la.ChunkConfig.from_env({**ENV, "NEMOTRON_CHUNK_MAX_SECONDS": "120", "NEMOTRON_CHUNK_TRIGGER_SECONDS": "120"})
    n = 300 * SR
    spans = la.plan_chunks(np.ones(n // 320, dtype=np.float32), n, cfg)
    assert 3 <= len(spans) <= 4 and all(e - s <= 120 * SR for s, e in spans)


# --- decode -------------------------------------------------------------------------------


def test_decode_resamples_stereo_and_finds_the_pause(tmp_path):
    audio, words = word_audio(n_words=40, pause_after={20: 3.0})
    src = tmp_path / "in.wav"
    write_wav(src, audio, rate=44_100, stereo=True)
    n, energy = la.decode_to_pcm(src, tmp_path / "raw", max_seconds=600)
    assert abs(n / SR - len(audio) / SR) < 0.05
    # The 3-second pause after word 20 is quiet in the energy track.
    pause_start = words[20][1]
    frames = energy[int((pause_start + 0.5) / la.FRAME_S) : int((pause_start + 2.5) / la.FRAME_S)]
    assert frames.max() < 1e-6 < energy.max()


def test_decode_and_split_writes_bounded_wavs(tmp_path):
    audio, words = word_audio(n_words=80, pause_after={25: 2.0, 50: 2.0})
    src = tmp_path / "in.m4a"
    write_m4a(src, audio)
    dur = la.probe_duration(src)
    assert dur is not None and abs(dur - len(audio) / SR) < 0.5
    cfg = la.ChunkConfig(max_seconds=30, trigger_seconds=30, overlap_seconds=2, search_seconds=6, max_audio_seconds=600)
    out = tmp_path / "chunks"
    out.mkdir()
    chunks = la.decode_and_split(str(src), str(out), cfg)
    assert len(chunks) >= 3
    assert not (out / "audio.s16").exists()
    for ch in chunks:
        with wave.open(ch.path) as w:
            assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (SR, 1, 2)
            assert w.getnframes() == ch.end - ch.start <= 30 * SR


def test_decode_rejects_audio_over_the_request_limit(tmp_path):
    audio, _ = word_audio(n_words=20)
    src = tmp_path / "in.wav"
    write_wav(src, audio)
    with pytest.raises(la.AudioInputError, match="NEMOTRON_MAX_AUDIO_SECONDS"):
        la.decode_to_pcm(src, tmp_path / "raw", max_seconds=5)


def test_garbage_is_a_client_error(tmp_path):
    src = tmp_path / "in.bin"
    src.write_bytes(b"not audio" * 100)
    with pytest.raises(la.AudioInputError):
        la.probe_duration(src)


# --- stitching ----------------------------------------------------------------------------


def test_stitch_drops_the_overlap_once():
    prev = "so we talked about the budget and then we moved on to hiring plans."
    nxt = "on to hiring plans. Next quarter we want three engineers."
    text, seam = la.stitch_pair(prev, nxt, 5)
    assert text == "so we talked about the budget and then we moved on to hiring plans. Next quarter we want three engineers."
    assert seam.matched == 4


def test_stitch_handles_partial_word_and_case_at_head():
    prev = "and that is why the Roadmap matters."
    # The next chunk began mid-word ("map,").
    text, seam = la.stitch_pair(prev, "map, Roadmap matters. Then we ship.", 5)
    assert text == "and that is why the Roadmap matters. Then we ship."
    # Joined at the middle matched word: prev keeps "Roadmap", next resumes at "matters.".
    assert seam == la.Seam(matched=2, dropped_prev=1, dropped_next=2)


def test_stitch_concatenates_when_overlap_was_silent():
    text, seam = la.stitch_pair("first part ends here.", "Second part starts.", 5)
    assert text == "first part ends here. Second part starts."
    assert seam.matched == 0


def test_stitch_single_word_only_at_the_edge():
    text, _ = la.stitch_pair("thank you everyone.", "Everyone. Let's start.", 5)
    assert text == "thank you Everyone. Let's start."
    # A lone shared word away from the edge is not trusted.
    text, seam = la.stitch_pair("the plan is ready now", "plan two is next", 5)
    assert seam.matched == 0 and text == "the plan is ready now plan two is next"


def test_stitch_texts_three_chunks_and_empty():
    text, seams = la.stitch_texts(["a b c d e", "d e f g h", "g h i j"], 5)
    assert text == "a b c d e f g h i j"
    assert [s.matched for s in seams] == [2, 2]
    assert la.stitch_texts(["", "x y"], 5)[0] == "x y"


# --- response merge -----------------------------------------------------------------------


def _resp(text, finish="stop", prompt=100, completion=10, ctc=None):
    r = {
        "id": "chatcmpl-x", "object": "chat.completion", "created": 1, "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text, "reasoning": None}, "finish_reason": finish}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion, "prompt_tokens_details": None},
    }
    if ctc is not None:
        r["ctc_timestamps"] = ctc
    return r


def test_merge_sums_usage_and_propagates_length():
    chunks = [la.Chunk(0, 0, 30 * SR), la.Chunk(1, 28 * SR, 50 * SR)]
    merged, _ = la.merge_chat_responses([_resp("a b c", "length", 100, 10), _resp("b c d", "stop", 50, 5)], chunks, 2)
    assert merged["choices"][0]["message"]["content"] == "a b c d"
    assert merged["choices"][0]["finish_reason"] == "length"
    assert merged["usage"] == {"prompt_tokens": 150, "completion_tokens": 15, "total_tokens": 165, "prompt_tokens_details": None}
    assert merged["object"] == "chat.completion" and merged["id"] == "chatcmpl-x"


def test_merge_moves_timestamps_onto_file_clock():
    chunks = [la.Chunk(0, 0, 30 * SR), la.Chunk(1, 28 * SR, 50 * SR)]
    w = lambda word, s, e: {"word": word, "start": s, "end": e, "speaker": "0"}  # noqa: E731
    c0 = {"words": [w("a", 1.0, 1.5), w("b", 29.0, 29.5)], "diarization": [{"speaker": 0, "start": 0.0, "end": 30.0}], "error": None}
    c1 = {"words": [w("b", 1.0, 1.5), w("c", 5.0, 5.5)], "diarization": [{"speaker": 1, "start": 0.0, "end": 22.0}], "error": None}
    merged, _ = la.merge_chat_responses([_resp("a b", ctc=c0), _resp("b c", ctc=c1)], chunks, 2)
    ctc = merged["ctc_timestamps"]
    # chunk 1's "b" at 28+1=29 s is in chunk 0's region (before its cut at 30 s): chunk 0's copy wins.
    assert [(x["word"], x["start"]) for x in ctc["words"]] == [("a", 1.0), ("b", 29.0), ("c", 33.0)]
    assert [(d["start"], d["end"]) for d in ctc["diarization"]] == [(0.0, 30.0), (30.0, 50.0)]
