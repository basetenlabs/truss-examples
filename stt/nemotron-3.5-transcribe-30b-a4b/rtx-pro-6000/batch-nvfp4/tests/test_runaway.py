"""data/runaway.py and the proxy's runaway guard, with synthetic runaway outputs.

The fake engine (tests/test_chunk_proxy.py) answers some requests the way the model ran
away in executions 11/12: invented fluent text at ~31 words/s up to max_tokens (the
TED-LIUM "YouTube vlog" chunk), or a phrase looping. Run as tests/test_long_audio.py says.
"""

from __future__ import annotations

import base64
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import runaway  # noqa: E402
from long_audio import ChunkConfig, split_in_two  # noqa: E402
from runaway import GuardConfig  # noqa: E402
from synth import read_words, word_audio, write_wav  # noqa: E402
from test_chunk_proxy import FakeEngine, _chat, _expected, _samples, _wav_bytes, rig  # noqa: E402,F401

LOOP = "I'll see you in the next video."


def _vlog(n: int) -> str:
    """Invented text that does not loop (the TED-LIUM chunk 2 failure)."""
    return " ".join(f"vlog{k}" for k in range(n))


def _data(audio) -> str:
    return "data:audio/wav;base64," + base64.b64encode(_wav_bytes(audio)).decode()


# --- NVIDIA's cleanup ---------------------------------------------------------------------


def test_cleanup_is_the_model_cards():
    assert runaway.cleanup_generation_loops("a" * 30) == "a"
    assert runaway.cleanup_generation_loops("ok " + "ha" * 25) == "ok ha"
    text = "Intro. " + " ".join([LOOP] * 25)
    out = runaway.cleanup_generation_loops(text)
    assert out.count("next video") <= 2 and out.startswith("Intro. ")
    # Under 20 copies is left alone.
    assert runaway.cleanup_generation_loops(" ".join([LOOP] * 15)) == " ".join([LOOP] * 15)


def test_fast_cleanup_matches_the_regex():
    rng = random.Random(0)
    vocab = [f"word{k}" for k in range(300)] + ["the", "a", "of", "and", "no", "yes,"]
    cases = [" ".join(rng.choice(vocab) for _ in range(800)) for _ in range(5)]
    for reps in (15, 18, 19, 20, 21, 40):
        for block in (LOOP, "no", "thank you,", "abc def", "x"):
            for sep in (" ", "", "\n"):
                cases.append("start here " + sep.join([block] * reps) + " end there")
    cases += ["a" * 25, "hello   " + " " * 30 + "x", "Supercalifragilisticexpialidocious ok", "", "\n" * 22]
    for c in cases:
        assert runaway.cleanup_if_looping(c) == runaway.cleanup_generation_loops(c), c[:80]


# --- detection ---------------------------------------------------------------------------


def test_longest_loop_flags_loops_not_speech():
    assert runaway.longest_loop("no, no, no, no, I said no") is None
    assert runaway.longest_loop("thank you thank you thank you so much") is None
    assert runaway.longest_loop(" ".join(f"w{k}" for k in range(2000))) is None
    sentence = "and that is why we went back to the shop"  # 10 words
    got = runaway.longest_loop("so " + " ".join([sentence] * 4) + " then")
    assert got is not None and got.period == 10 and got.repeats == 4 and got.start == 1
    got = runaway.longest_loop("x " + " ".join([LOOP] * 9))
    assert got is not None and got.period == 7 and got.repeats == 9
    assert runaway.longest_loop(" ".join(["the"] * 17)) is not None
    # Case and punctuation do not hide a loop.
    assert runaway.longest_loop(" ".join(["Bye.", "bye"] * 10)) is not None


def test_assess_rules():
    g = GuardConfig()
    speech = " ".join(f"w{k}" for k in range(150))  # 2.5 words/s over 60 s
    budget = runaway.token_budget(60, g)
    assert not runaway.assess(speech, "stop", 60, budget, g).anomalous
    # Hitting our duration budget is a runaway; hitting a smaller client limit is not.
    assert runaway.assess(speech, "length", 60, budget, g).reasons == ["length"]
    assert not runaway.assess(speech, "length", 60, 100, g).anomalous
    # Over 8 words/s absolute (31/s in the TED-LIUM chunk).
    a = runaway.assess(_vlog(60 * 31), "stop", 60, budget, g)
    assert a.anomalous and a.reasons[0].startswith("rate>8") and a.words_per_s == pytest.approx(31)
    # A 1-second "yes, okay, sure" is within the slack.
    assert not runaway.assess("yes, okay, sure, right, fine", "stop", 1, budget, g).anomalous
    # 6 words/s against a median of 2.0: relative rule; not without a median.
    six = _vlog(360)
    assert "median" in runaway.assess(six, "stop", 60, budget, g, median_wps=2.0).reasons[0]
    assert not runaway.assess(six, "stop", 60, budget, g, median_wps=3.0).anomalous
    assert not runaway.assess(six, "stop", 60, budget, g).anomalous
    assert runaway.median([1.0, 2.0]) is None and runaway.median([1.0, 2.0, 30.0]) == 2.0
    assert runaway.assess(speech + " " + " ".join([LOOP] * 9), "stop", 60, budget, g).loop is not None


def test_token_cap():
    g = GuardConfig()
    assert runaway.token_budget(3600, g) == 3600 * 8 + 256
    b = {}
    assert runaway.apply_token_cap(b, 10, g) == 336 and b == {"max_tokens": 336}
    b = {"max_tokens": 16000}
    assert runaway.apply_token_cap(b, 600, g) == 5056 and b["max_tokens"] == 5056
    b = {"max_tokens": 100}
    assert runaway.apply_token_cap(b, 600, g) == 100 and b["max_tokens"] == 100
    b = {"max_completion_tokens": 9000, "max_tokens": 16000}
    assert runaway.apply_token_cap(b, 1800, g) == 9000 and b == {"max_completion_tokens": 9000, "max_tokens": 9000}


def test_guard_config_from_env():
    g = GuardConfig.from_env({})
    assert (g.enabled, g.tokens_per_s, g.margin_tokens, g.split_min_seconds) == (True, 8.0, 256, 600.0)
    assert not GuardConfig.from_env({"NEMOTRON_RUNAWAY_GUARD": "0"}).enabled
    assert GuardConfig.from_env({"NEMOTRON_RUNAWAY_TOKENS_PER_S": "6"}).tokens_per_s == 6.0
    for bad in ({"NEMOTRON_RUNAWAY_GUARD": "maybe"}, {"NEMOTRON_RUNAWAY_TOKENS_PER_S": "0.5"},
                {"NEMOTRON_RUNAWAY_MARGIN_TOKENS": "-1"}, {"NEMOTRON_RUNAWAY_SPLIT_MIN_SECONDS": "x"}):
        with pytest.raises(ValueError):
            GuardConfig.from_env(bad)


def test_split_in_two_cuts_in_a_pause(tmp_path):
    audio, words = word_audio(n_words=40, pause_after={19: 1.5})
    write_wav(tmp_path / "a.wav", audio)
    cfg = ChunkConfig(max_seconds=30, trigger_seconds=36, overlap_seconds=2, search_seconds=6)
    halves = split_in_two(str(tmp_path / "a.wav"), str(tmp_path / "out"), cfg, offset=16000 * 100)
    assert len(halves) == 2
    a, b = halves
    assert a.start_s == 100 and b.end == 16000 * 100 + len(audio)
    assert a.end - b.start == 2 * 16000  # 2 s overlap
    # The cut is in a silent gap near the middle, so no word is cut in two.
    cut = a.end_s - 100
    assert 15 < cut < 26 and not any(s < cut < e for s, e in words)
    wa = read_words(_samples(Path(a.path).read_bytes()))
    wb = read_words(_samples(Path(b.path).read_bytes()))
    assert wa == [f"w{k}" for k in range(len(wa))] and wb[-1] == "w39"
    assert set(wa) | set(wb) == {f"w{k}" for k in range(40)}


# --- the proxy ---------------------------------------------------------------------------

GUARD = GuardConfig(split_min_seconds=20, chunk_split_min_seconds=20)


async def test_short_request_gets_a_duration_cap(rig):
    audio, _ = word_audio(n_words=10)
    engine = FakeEngine()
    client, _ = await rig(engine, guard=GUARD)
    r = await client.post("/v1/chat/completions", json=_chat(_data(audio), max_tokens=16000))
    assert r.status == 200 and "X-Transcript-Guard" not in r.headers and "X-Audio-Chunks" not in r.headers
    assert (await r.json())["choices"][0]["message"]["content"] == _expected(10)
    assert engine.requests[0]["max_tokens"] == runaway.token_budget(len(audio) / 16000, GUARD)
    assert engine.requests[0]["messages"][0]["content"][1]["audio_url"]["url"].startswith("file://")


async def test_thinking_requests_pass_through(rig):
    audio, _ = word_audio(n_words=10)
    engine = FakeEngine()
    client, _ = await rig(engine, guard=GUARD)
    raw = json.dumps(_chat(_data(audio), chat_template_kwargs={"enable_thinking": True})).encode()
    r = await client.post("/v1/chat/completions", data=raw, headers={"Content-Type": "application/json"})
    assert r.status == 200 and engine.raw == [raw]


async def test_unchunked_runaway_is_retried_as_halves(rig, caplog):
    audio, _ = word_audio(n_words=33)  # ~33 s: one engine request (trigger 36 s)

    def runaway_once(idx, words, dur, body):
        if idx == 0:  # the whole file: the real words, then a vlog up to max_tokens
            return " ".join(words) + " " + _vlog(body["max_tokens"]), "length"
        return None

    engine = FakeEngine(hallucinate=runaway_once)
    client, _ = await rig(engine, guard=GUARD)
    with caplog.at_level("DEBUG", logger="chunk_proxy"):
        r = await client.post("/v1/chat/completions", json=_chat(_data(audio)))
    body = await r.json()
    assert r.status == 200
    assert body["choices"][0]["message"]["content"] == _expected(33)
    assert body["choices"][0]["finish_reason"] == "stop"
    assert r.headers["X-Transcript-Guard"] == "flagged=1; retried=1; fixed=1; anomalous=0; cleaned=0"
    assert len(engine.requests) == 3
    assert engine.requests[0]["max_tokens"] == runaway.token_budget(len(audio) / 16000, GUARD)
    msgs = [rec.getMessage() for rec in caplog.records]
    assert any("anomalous: length; rate>8/s" in m for m in msgs)
    assert any("retried as" in m and "['ok', 'ok']" in m for m in msgs)


async def test_failed_retry_is_logged_without_the_error_body(rig, caplog):
    audio, _ = word_audio(n_words=33)  # one engine request, retried as two halves

    def runaway_once(idx, words, dur, body):
        if idx == 0:
            return " ".join(words) + " " + _vlog(body["max_tokens"]), "length"
        return None

    engine = FakeEngine(hallucinate=runaway_once, fail_on_chunk=1)  # the first half's retry fails
    client, _ = await rig(engine, guard=GUARD)
    with caplog.at_level("DEBUG", logger="chunk_proxy"):
        r = await client.post("/v1/chat/completions", json=_chat(_data(audio)))
    assert r.status == 200
    failed = [m for m in (rec.getMessage() for rec in caplog.records) if "retry failed" in m]
    assert len(failed) == 1 and "status=400 body_bytes=" in failed[0]
    assert "boom" not in caplog.text  # ZDR: the engine's error body never reaches the log


async def test_chunked_runaway_chunk_is_retried(rig, caplog):
    audio, _ = word_audio(n_words=90, pause_after={27: 1.5, 55: 1.5})

    def vlog_in_last_chunk(idx, words, dur, body):
        if "w80" in words and dur > 22:  # the full last chunk, not its retried halves
            return " ".join(words) + " " + _vlog(int(dur * 31)), "length"
        return None

    engine = FakeEngine(hallucinate=vlog_in_last_chunk)
    client, _ = await rig(engine, guard=GUARD)
    with caplog.at_level("DEBUG", logger="chunk_proxy"):
        r = await client.post("/v1/chat/completions", json=_chat(_data(audio)))
    body = await r.json()
    n = int(r.headers["X-Audio-Chunks"])
    assert body["choices"][0]["message"]["content"] == _expected(90)
    assert body["choices"][0]["finish_reason"] == "stop"
    assert r.headers["X-Transcript-Guard"].startswith("flagged=1; retried=1; fixed=1;")
    assert len(engine.requests) == n + 1  # n - 1 original chunks + the bad one + 2 halves
    assert any("runaway guard" in rec.getMessage() and "anomalous" in rec.getMessage() for rec in caplog.records)


async def test_relative_rate_flags_a_chunk(rig):
    audio, _ = word_audio(n_words=90, pause_after={27: 1.5, 55: 1.5})

    def invented_tail(idx, words, dur, body):
        if "w40" in words and dur > 22:  # 6 words/s against ~1/s elsewhere, finish=stop
            return " ".join(words) + " " + _vlog(int(dur * 5)), "stop"
        return None

    engine = FakeEngine(hallucinate=invented_tail)
    client, _ = await rig(engine, guard=GUARD)
    r = await client.post("/v1/chat/completions", json=_chat(_data(audio)))
    body = await r.json()
    assert body["choices"][0]["message"]["content"] == _expected(90)
    assert r.headers["X-Transcript-Guard"].startswith("flagged=1; retried=1; fixed=1;")


async def test_persistent_loop_is_cleaned_and_marked_length(rig, caplog):
    audio, _ = word_audio(n_words=33)

    def always_loop(idx, words, dur, body):
        return " ".join(words) + " " + " ".join([LOOP] * 40), "stop"

    engine = FakeEngine(hallucinate=always_loop)
    client, _ = await rig(engine, guard=GUARD)
    with caplog.at_level("DEBUG", logger="chunk_proxy"):
        r = await client.post("/v1/chat/completions", json=_chat(_data(audio)))
    body = await r.json()
    text = body["choices"][0]["message"]["content"]
    assert body["choices"][0]["finish_reason"] == "length"
    assert r.headers["X-Transcript-Guard"] == "flagged=1; retried=1; fixed=0; anomalous=1; cleaned=1"
    assert len(engine.requests) == 3
    # Both halves still looped, so the original is kept, with the loop collapsed.
    assert text.startswith(_expected(33)) and text.count("next video") <= 2
    assert any("still anomalous" in rec.getMessage() for rec in caplog.records)


async def test_short_runaway_is_not_split_but_cleaned(rig):
    audio, _ = word_audio(n_words=10)  # under split_min_seconds: no retry

    def loop(idx, words, dur, body):
        return " ".join(words) + " " + " ".join(["Bye."] * 60), "length"

    engine = FakeEngine(hallucinate=loop)
    client, _ = await rig(engine, guard=GUARD)
    r = await client.post("/v1/chat/completions", json=_chat(_data(audio), stream=True))
    assert r.headers["Content-Type"].startswith("text/event-stream")
    assert r.headers["X-Transcript-Guard"] == "flagged=1; retried=0; fixed=0; anomalous=1; cleaned=1"
    events = [json.loads(line[6:]) for line in (await r.text()).split("\n\n") if line.startswith("data: {")]
    assert events[0]["choices"][0]["delta"]["content"] == _expected(10) + " Bye."
    assert events[1]["choices"][0]["finish_reason"] == "length"
    assert len(engine.requests) == 1 and engine.requests[0]["stream"] is False


async def test_guard_off_leaves_runaways_alone(rig):
    audio, _ = word_audio(n_words=33)
    engine = FakeEngine(hallucinate=lambda i, w, d, b: (" ".join(w) + " " + _vlog(500), "length"))
    client, _ = await rig(engine, guard=GuardConfig(enabled=False))
    r = await client.post("/v1/chat/completions", json=_chat(_data(audio)))
    assert "X-Transcript-Guard" not in r.headers and len(engine.requests) == 1
    assert engine.requests[0]["max_tokens"] == 1000


def test_loop_detection_is_fast_on_a_long_runaway():
    import time

    text = _vlog(3000) + " " + " ".join([LOOP] * 1500)  # ~13.5k words, as the TED chunk
    t0 = time.monotonic()
    assert runaway.longest_loop(text) is not None
    cleaned = runaway.cleanup_if_looping(_vlog(12000))
    assert time.monotonic() - t0 < 5 and cleaned == _vlog(12000)
    assert np.isfinite(runaway.words_per_s(text, 400))
