"""CPU-only tests for the vv_longform proxy against a fake vLLM upstream.

Run from the preset dir: ``PYTHONPATH=packages python3 -m pytest packages/vv_longform/tests -q``
(needs aiohttp, numpy, pytest and an ffmpeg binary).
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import shutil
import subprocess
from dataclasses import replace

import numpy as np
import pytest
from aiohttp import FormData, web
from aiohttp.test_utils import TestServer

from vv_longform import audio, chunking, transcript
from vv_longform.config import Settings, load
from vv_longform.server import Proxy, canonical_text

# The proxy's defaults are the shipped preset (chunking, prompt rewrite, loop guard). Most tests start
# from the pre-proxy behaviour and turn on what they test.
OFF = {"chunk_s": 0.0, "canonical_prompt": False, "max_tokens_per_s": None, "prep_parallel": False,
       "max_model_len": 32768}  # fmt: skip

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")


def tone(seconds: float, rate: int = 16000) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    x = 0.3 * np.sin(2 * np.pi * 220 * t)
    x[(t % 10) > 9] = 0.0  # one second of silence every 10 s
    return (x * 32767).astype(np.int16)


class FakeVLLM:
    """Records chat bodies and replies like VibeVoice: one segment per request."""

    def __init__(self):
        self.bodies: list[dict] = []

    def app(self) -> web.Application:
        app = web.Application(client_max_size=1 << 30)
        app.router.add_get("/v1/models", self.models)
        app.router.add_post("/v1/chat/completions", self.chat)
        app.router.add_get("/clip.m4a", self.clip)
        return app

    async def models(self, request):
        return web.json_response({"data": [{"id": "vibevoice"}]})

    async def clip(self, request):
        return web.Response(body=request.app["m4a"], content_type="audio/mp4")

    async def chat(self, request):
        body = await request.json()
        self.bodies.append(body)
        url = next(
            p["audio_url"]["url"]
            for m in body["messages"] if isinstance(m.get("content"), list)
            for p in m["content"] if p.get("type") == "audio_url"
        )  # fmt: skip
        _, data = audio.split_data_uri(url)
        dur = audio.wav_duration_s(data)
        content = json.dumps([
            {"Start": 0.0, "End": round(dur / 2, 2), "Speaker": 0, "Content": "hello"},
            {"Start": round(dur / 2, 2), "End": round(dur, 2), "Speaker": 1, "Content": "world"},
        ])  # fmt: skip
        return web.json_response({
            "id": "x", "object": "chat.completion", "model": "vibevoice",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": int(dur * 7.5), "completion_tokens": 20, "total_tokens": 0},
        })  # fmt: skip


def run(coro):
    return asyncio.run(coro)


async def _with_servers(settings_kw, fn, m4a: bytes = b""):
    fake = FakeVLLM()
    up_app = fake.app()
    up_app["m4a"] = m4a
    up = TestServer(up_app)
    await up.start_server()
    s = replace(Settings(), upstream=str(up.make_url("")).rstrip("/"), **{**OFF, **settings_kw})
    proxy = TestServer(Proxy(s).app())
    await proxy.start_server()
    import aiohttp

    try:
        async with aiohttp.ClientSession() as client:
            return await fn(client, str(proxy.make_url("")).rstrip("/"), fake, str(up.make_url("")).rstrip("/"))
    finally:
        await proxy.close()
        await up.close()


def chat_body(url: str, max_tokens: int = 500) -> dict:
    return {
        "model": "vibevoice",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Transcribe this audio."},
            {"type": "audio_url", "audio_url": {"url": url}},
        ]}],
        "temperature": 0, "max_tokens": max_tokens,
    }  # fmt: skip


def wav_uri(pcm, rate=16000) -> str:
    return "data:audio/wav;base64," + base64.b64encode(audio.to_wav(pcm, rate)).decode()


def test_defaults_match_config_yaml(monkeypatch):
    """With no VV_* set, the proxy runs the shipped configuration (config.yaml's env block)."""
    import os
    import pathlib

    cfg = pathlib.Path(__file__).resolve().parents[3] / "config.yaml"
    env = dict(re.findall(r'^\s+(VV_[A-Z_]+):\s+"([^"]*)"', cfg.read_text(), re.MULTILINE))
    for k in list(os.environ):
        if k.startswith("VV_"):
            monkeypatch.delenv(k)
    s = load()
    assert s == Settings()
    assert s.upstream == "http://127.0.0.1:8001" and s.listen_port == 8000
    assert s.max_model_len == int(env["VV_MAX_MODEL_LEN"])
    assert s.canonical_prompt == (env["VV_CANONICAL_PROMPT"] == "1")
    assert s.max_tokens_per_s == float(env["VV_MAX_TOKENS_PER_S"])
    assert s.max_tokens_base == int(env["VV_MAX_TOKENS_BASE"])
    assert s.max_tokens_cap == int(env["VV_MAX_TOKENS_CAP"])
    assert s.chunk_s == float(env["VV_CHUNK_S"])
    assert s.speaker_link == env["VV_LINK"] and s.link_threshold == float(env["VV_LINK_THRESHOLD"])
    assert s.prep_parallel == (env["VV_PREP_PARALLEL"] == "1")
    monkeypatch.setenv("VV_MAX_TOKENS_PER_S", "0")
    assert load().max_tokens_per_s is None


def test_models_passthrough():
    async def fn(client, base, fake, _):
        async with client.get(base + "/v1/models") as r:
            assert r.status == 200
            assert (await r.json())["data"][0]["id"] == "vibevoice"

    run(_with_servers({}, fn))


def test_chat_wav_passthrough_unchanged(capsys):
    body = chat_body(wav_uri(tone(12)))

    async def fn(client, base, fake, _):
        async with client.post(base + "/v1/chat/completions", json=body) as r:
            assert r.status == 200
            out = await r.json()
        assert fake.bodies == [body]  # byte-for-byte same request
        assert json.loads(out["choices"][0]["message"]["content"])[0]["Content"] == "hello"

    run(_with_servers({}, fn))
    line = [x for x in capsys.readouterr().out.splitlines() if x.startswith("timing ")][0]
    assert "route=chat" in line and "audio_s=12.0" in line and "n_chunks=1" in line
    assert "finish_reason=stop" in line and "completion_tokens=20" in line


@needs_ffmpeg
def test_chat_m4a_url_is_decoded(tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(audio.to_wav(tone(8), 16000))
    m4a = tmp_path / "a.m4a"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(wav), "-c:a", "aac",
                    "-movflags", "+faststart", str(m4a)], check=True)  # fmt: skip

    async def fn(client, base, fake, up):
        body = chat_body(up + "/clip.m4a")
        async with client.post(base + "/v1/chat/completions", json=body) as r:
            assert r.status == 200, await r.text()
        sent = fake.bodies[0]["messages"][0]["content"][1]["audio_url"]["url"]
        assert sent.startswith("data:audio/wav;base64,")
        dur = audio.wav_duration_s(audio.split_data_uri(sent)[1])
        assert abs(dur - 8) < 0.2
        assert fake.bodies[0]["max_tokens"] == 500  # unchanged by default

    run(_with_servers({}, fn, m4a=m4a.read_bytes()))


@needs_ffmpeg
def test_transcriptions_verbose_json():
    flac = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "s16le", "-ar", "16000", "-ac", "1", "-i", "-",
         "-f", "flac", "-"], input=tone(30).tobytes(), capture_output=True, check=True,
    ).stdout  # fmt: skip

    async def fn(client, base, fake, _):
        form = FormData()
        form.add_field("model", "vibevoice")
        form.add_field("response_format", "verbose_json")
        form.add_field("file", flac, filename="audio.flac", content_type="audio/flac")
        async with client.post(base + "/v1/audio/transcriptions", data=form) as r:
            assert r.status == 200, await r.text()
            out = await r.json()
        assert [s["speaker"] for s in out["segments"]] == ["S0", "S1"]
        assert out["segments"][1]["end"] == pytest.approx(30, abs=0.1)
        assert len(fake.bodies) == 1
        assert fake.bodies[0]["max_tokens"] == int(20 * 30 + 200)  # README rule

    run(_with_servers({}, fn))


@needs_ffmpeg
def test_transcriptions_chunked_and_levers(capsys):
    flac = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "s16le", "-ar", "16000", "-ac", "1", "-i", "-",
         "-f", "flac", "-"], input=tone(95).tobytes(), capture_output=True, check=True,
    ).stdout  # fmt: skip
    kw = {"chunk_s": 30.0, "cut_search_s": 5.0, "max_tokens_per_s": 9.0, "canonical_prompt": True,
          "speaker_link": "none"}

    async def fn(client, base, fake, _):
        form = FormData()
        form.add_field("model", "vibevoice")
        form.add_field("response_format", "verbose_json")
        form.add_field("file", flac, filename="audio.flac", content_type="audio/flac")
        async with client.post(base + "/v1/audio/transcriptions", data=form) as r:
            assert r.status == 200, await r.text()
            out = await r.json()
        assert len(fake.bodies) == 4  # cuts near 29, 59, 89 s; 6 s tail kept
        b = fake.bodies[0]
        assert b["messages"][0]["role"] == "system"
        assert "seconds audio" in b["messages"][1]["content"][-1]["text"]
        segs = out["segments"]
        assert segs[-1]["end"] == pytest.approx(95, abs=0.1)
        assert all(segs[i]["start"] <= segs[i + 1]["start"] for i in range(len(segs) - 1))
        assert {s["speaker"] for s in segs} >= {"c0s0", "c2s1"}

    run(_with_servers(kw, fn))
    line = [x for x in capsys.readouterr().out.splitlines() if x.startswith("timing ")][0]
    assert "n_chunks=4" in line


def test_chunk_bounds_cover_without_overlap():
    pcm = tone(125)
    b = chunking.bounds(pcm, 16000, 30, 5)
    assert b[0][0] == 0 and b[-1][1] == len(pcm)
    assert all(b[i][1] == b[i + 1][0] for i in range(len(b) - 1))
    # cuts land on the silent second inside each 10 s period
    for _, end in b[:-1]:
        assert (end / 16000) % 10 > 8.9
    assert chunking.bounds(pcm, 16000, 0, 5) == [(0, len(pcm))]


def test_parse_truncated_reply():
    text = '[{"Start": 0, "End": 1.5, "Speaker": 0, "Content": "a"}, {"Start": 1.5, "End": 3, "Spe'
    segs = transcript.parse(text)
    assert len(segs) == 1 and segs[0].text == "a" and segs[0].speaker == "0"
    assert transcript.parse("not json") == []


# The upstream recipe, copied verbatim from microsoft/VibeVoice@16fb2cb (vllm_plugin/tests/test_api.py:150-175,
# vibevoice/processor/vibevoice_asr_processor.py:27,396). Kept as literals so a proxy edit cannot drift.
UPSTREAM_SYSTEM = "You are a helpful assistant that transcribes audio input into text output in JSON format."


def upstream_text(duration: float) -> str:
    show_keys = ["Start time", "End time", "Speaker ID", "Content"]
    return (f"This is a {duration:.2f} seconds audio, please transcribe it with these keys: "
            + ", ".join(show_keys))  # fmt: skip


def assert_upstream_shape(body: dict, duration: float) -> None:
    msgs = body["messages"]
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == UPSTREAM_SYSTEM
    parts = msgs[1]["content"]
    assert [p["type"] for p in parts] == ["audio_url", "text"]
    assert parts[1]["text"] == upstream_text(duration)
    assert "stop_token_ids" not in body  # stop tokens come from generation_config.json (data/start.sh)


def test_canonical_prompt_chat_matches_upstream():
    body = chat_body(wav_uri(tone(12)), max_tokens=600)
    body["messages"].insert(0, {"role": "system", "content": "client system prompt"})
    kw = {"canonical_prompt": True, "max_tokens_per_s": 20.0, "max_tokens_base": 200}

    async def fn(client, base, fake, _):
        async with client.post(base + "/v1/chat/completions", json=body) as r:
            assert r.status == 200, await r.text()
        sent = fake.bodies[0]
        assert_upstream_shape(sent, 12.0)
        assert upstream_text(12.0) == ("This is a 12.00 seconds audio, please transcribe it with these keys: "
                                       "Start time, End time, Speaker ID, Content")  # fmt: skip
        assert sent["max_tokens"] == 20 * 12 + 200  # the guard lowers the client's 600

    run(_with_servers(kw, fn))


@pytest.mark.parametrize("client, expected", [(100, 100), (100.0, 100), (5000, 20 * 12 + 200),
                                               (None, 20 * 12 + 200), ("lots", 20 * 12 + 200),
                                               (True, 20 * 12 + 200), (0, 20 * 12 + 200)])  # fmt: skip
def test_loop_guard_only_lowers_client_max_tokens(client, expected):
    body = chat_body(wav_uri(tone(12)))
    if client is None:
        del body["max_tokens"]
    else:
        body["max_tokens"] = client
    kw = {"canonical_prompt": True, "max_tokens_per_s": 20.0, "max_tokens_base": 200}

    async def fn(client_, base, fake, _):
        async with client_.post(base + "/v1/chat/completions", json=body) as r:
            assert r.status == 200, await r.text()
        assert fake.bodies[0]["max_tokens"] == expected

    run(_with_servers(kw, fn))


# Upstream's hotword form (vibevoice_asr_processor.py:393-394 at 16fb2cb), kept literal.
UPSTREAM_HOTWORDS = ("This is a 12.00 seconds audio, with extra info: Baseten, VibeVoice, tcpWER\n\n"
                     "Please transcribe it with these keys: Start time, End time, Speaker ID, Content")  # fmt: skip


def test_canonical_prompt_keeps_hotwords():
    assert canonical_text(12.0, "Baseten, VibeVoice, tcpWER") == UPSTREAM_HOTWORDS
    assert canonical_text(12.0, "  ") == upstream_text(12.0)
    body = chat_body(wav_uri(tone(12)))
    # A client that sends upstream's own recipe text, hotwords included.
    body["messages"][0]["content"][0]["text"] = UPSTREAM_HOTWORDS

    async def fn(client, base, fake, _):
        async with client.post(base + "/v1/chat/completions", json=body) as r:
            assert r.status == 200, await r.text()
        assert fake.bodies[0]["messages"][1]["content"][-1]["text"] == UPSTREAM_HOTWORDS

    run(_with_servers({"canonical_prompt": True}, fn))


def test_hotwords_reach_every_chunk():
    body = chat_body(wav_uri(tone(95)))
    body["messages"][0]["content"][0]["text"] = "Transcribe. with extra info: Baseten, VibeVoice"

    async def fn(client, base, fake, _):
        async with client.post(base + "/v1/chat/completions", json=body) as r:
            assert r.status == 200, await r.text()
            out = await r.json()
        assert len(fake.bodies) == 4
        for b in fake.bodies:
            assert "with extra info: Baseten, VibeVoice\n\nPlease transcribe" in b["messages"][1]["content"][-1]["text"]
        # Chunk-local labels stay strings; numeric labels come back as ints (test_speaker_ints below).
        assert {type(r["Speaker"]) for r in json.loads(out["choices"][0]["message"]["content"])} == {str}

    run(_with_servers({"canonical_prompt": True, "chunk_s": 30.0, "cut_search_s": 5.0,
                       "speaker_link": "none"}, fn))  # fmt: skip


def test_speaker_ints_in_stitched_reply():
    segs = [transcript.Segment(0, 1, "0", "a"), transcript.Segment(1, 2, "12", "b"),
            transcript.Segment(2, 3, "unknown", "c"), transcript.Segment(3, 4, "c1s0", "d")]  # fmt: skip
    rows = json.loads(transcript.to_model_json(segs))
    assert [r["Speaker"] for r in rows] == [0, 12, "unknown", "c1s0"]


@needs_ffmpeg
def test_canonical_prompt_transcriptions_matches_upstream():
    flac = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "s16le", "-ar", "16000", "-ac", "1", "-i", "-",
         "-f", "flac", "-"], input=tone(30).tobytes(), capture_output=True, check=True,
    ).stdout  # fmt: skip

    async def fn(client, base, fake, _):
        form = FormData()
        form.add_field("model", "vibevoice")
        form.add_field("response_format", "verbose_json")
        form.add_field("file", flac, filename="audio.flac", content_type="audio/flac")
        async with client.post(base + "/v1/audio/transcriptions", data=form) as r:
            assert r.status == 200, await r.text()
        assert len(fake.bodies) == 1
        assert_upstream_shape(fake.bodies[0], 30.0)

    run(_with_servers({"canonical_prompt": True}, fn))


def test_canonical_off_adds_no_stop_tokens():
    body = chat_body(wav_uri(tone(5)))

    async def fn(client, base, fake, _):
        async with client.post(base + "/v1/chat/completions", json=body) as r:
            assert r.status == 200
        assert fake.bodies == [body] and "stop_token_ids" not in fake.bodies[0]

    run(_with_servers({}, fn))


# --- stale keep-alive connections (batch-bench run 44176947016: 503 "[Errno 32] Broken pipe" /
# "Server disconnected" before generation) ------------------------------------------------------


async def _stale_servers(fn, *, up_keepalive: float | None = None, drop_reused: bool = False,
                         settings_kw: dict | None = None, m4a: bytes = b""):
    """Like ``_with_servers`` but the fake upstream can close idle keep-alive connections after
    ``up_keepalive`` s (uvicorn's timeout_keep_alive; vLLM default 5 s) or drop every request that
    arrives on a reused connection without answering (the close/reuse race, made deterministic)."""
    fake = FakeVLLM()
    up_app = fake.app()
    up_app["m4a"] = m4a
    seen: dict[tuple, int] = {}  # requests per client (host, port); id(transport) gets recycled

    @web.middleware
    async def drop(request, handler):
        key = request.transport.get_extra_info("peername")
        seen[key] = seen.get(key, 0) + 1
        if drop_reused and seen[key] > 1:
            await request.read()
            request.transport.close()  # FIN without a response, like a keep-alive close
            raise asyncio.CancelledError
        return await handler(request)

    up_app.middlewares.append(drop)
    # The upstream gets its own thread and event loop, as vLLM is its own process: its keep-alive
    # timer must keep running while the proxy's loop is blocked.
    import threading

    up_loop = asyncio.new_event_loop()
    started = threading.Event()
    box: dict = {}

    async def serve():
        runner = web.AppRunner(up_app, keepalive_timeout=up_keepalive if up_keepalive is not None else 75.0)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        box["runner"] = runner
        box["port"] = runner.addresses[0][1]
        started.set()

    th = threading.Thread(target=lambda: (up_loop.run_until_complete(serve()), up_loop.run_forever()),
                          daemon=True)  # fmt: skip
    th.start()
    started.wait(10)
    up_url = f"http://127.0.0.1:{box['port']}"
    s = replace(Settings(), upstream=up_url, **{**OFF, **(settings_kw or {})})
    proxy = Proxy(s)
    srv = TestServer(proxy.app())
    await srv.start_server()
    import aiohttp

    try:
        async with aiohttp.ClientSession() as client:
            return await fn(client, str(srv.make_url("")).rstrip("/"), fake, up_url, proxy)
    finally:
        await srv.close()
        asyncio.run_coroutine_threadsafe(box["runner"].cleanup(), up_loop).result(10)
        up_loop.call_soon_threadsafe(up_loop.stop)
        th.join(10)


@pytest.mark.parametrize("pool_keepalive", [0.2, None])
def test_upstream_idle_close_while_proxy_loop_busy(monkeypatch, capsys, pool_keepalive):
    """Root cause: the proxy's event loop is blocked in synchronous prep (WAV probe/encode,
    json.dumps of a multi-MB body) while vLLM closes the idle pooled connection, so the FIN is
    not seen before the connection is reused. The pool must expire connections by age before
    vLLM's keep-alive does, and a stale connection must be retried once on a fresh one."""
    import time as _time

    body = chat_body(wav_uri(tone(12)))
    real = audio.wav_duration_s
    block = {"s": 0.0}

    def slow_probe(data):
        _time.sleep(block["s"])  # stands in for seconds of synchronous prep on long audio
        return real(data)

    monkeypatch.setattr(audio, "wav_duration_s", slow_probe)

    async def fn(client, base, fake, _up, proxy):
        async with client.post(base + "/v1/chat/completions", json=body) as r:
            assert r.status == 200
        block["s"] = 0.6  # longer than the upstream keep-alive below
        async with client.post(base + "/v1/chat/completions", json=body) as r:
            assert r.status == 200, await r.text()
        assert len(fake.bodies) == 2

    if pool_keepalive is not None:  # pool expiry below the upstream's keep-alive: no stale reuse
        from vv_longform import server

        monkeypatch.setattr(server, "UPSTREAM_KEEPALIVE_S", pool_keepalive)
    run(_stale_servers(fn, up_keepalive=0.3))
    lines = [x for x in capsys.readouterr().out.splitlines() if x.startswith("timing ")]
    assert len(lines) == 2 and all("status=200" in x for x in lines)
    # Without the expiry the second request reuses the closed connection and needs the retry.
    assert ("upstream_retries=1" in lines[1]) == (pool_keepalive is None)


def test_retry_once_on_stale_pooled_upstream_connection(capsys):
    body = chat_body(wav_uri(tone(12)))

    async def fn(client, base, fake, _up, proxy):
        for _ in range(3):
            async with client.post(base + "/v1/chat/completions", json=body) as r:
                assert r.status == 200, await r.text()
        async with client.get(base + "/v1/models") as r:
            assert r.status == 200
        assert len(fake.bodies) == 3

    run(_stale_servers(fn, drop_reused=True))
    lines = [x for x in capsys.readouterr().out.splitlines() if x.startswith("timing ")]
    assert all("status=200" in x for x in lines)
    assert sum("upstream_retries=1" in x for x in lines) >= 1


@needs_ffmpeg
def test_retry_once_on_stale_pooled_upstream_connection_chunked(tmp_path):
    flac = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "s16le", "-ar", "16000", "-ac", "1", "-i", "-",
         "-f", "flac", "-"], input=tone(30).tobytes(), capture_output=True, check=True,
    ).stdout  # fmt: skip

    async def fn(client, base, fake, _up, proxy):
        for _ in range(2):
            form = FormData()
            form.add_field("file", flac, filename="audio.flac", content_type="audio/flac")
            async with client.post(base + "/v1/audio/transcriptions", data=form) as r:
                assert r.status == 200, await r.text()
        assert len(fake.bodies) == 2

    run(_stale_servers(fn, drop_reused=True))


@needs_ffmpeg
def test_remote_audio_fetch_drop_is_retried_then_502(tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(audio.to_wav(tone(4), 16000))
    m4a = tmp_path / "a.m4a"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(wav), "-c:a", "aac",
                    "-movflags", "+faststart", str(m4a)], check=True)  # fmt: skip

    async def ok(client, base, fake, up, proxy):
        for _ in range(3):  # fetches and engine posts both survive reused-connection drops
            async with client.post(base + "/v1/chat/completions", json=chat_body(up + "/clip.m4a")) as r:
                assert r.status == 200, await r.text()

    run(_stale_servers(ok, drop_reused=True, m4a=m4a.read_bytes()))

    async def dead(client, base, fake, up, proxy):
        # Port 9 (discard) refuses: the remote host is down, not the engine.
        async with client.post(base + "/v1/chat/completions", json=chat_body("http://127.0.0.1:9/a.m4a")) as r:
            assert r.status == 502
            assert "audio fetch failed" in (await r.json())["error"]["message"]

    run(_stale_servers(dead))
