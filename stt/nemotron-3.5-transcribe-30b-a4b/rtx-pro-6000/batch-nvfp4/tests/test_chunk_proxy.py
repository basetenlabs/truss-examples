"""data/chunk_proxy.py against a fake engine: routing, chunking, stitching, response format.

The fake engine stands in for vLLM's /v1/chat/completions: it reads the audio it is
given (data URI or the proxy's file:// chunk), "transcribes" each synthetic word burst
(tests/synth.py) as ``w<k>``, and answers in vLLM's chat.completion format. Run as
tests/test_long_audio.py says.
"""

from __future__ import annotations

import base64
import concurrent.futures
import io
import json
import sys
import wave
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import chunk_proxy  # noqa: E402
from long_audio import ChunkConfig  # noqa: E402
from runaway import GuardConfig, token_budget  # noqa: E402
from synth import read_words, word_audio, write_wav  # noqa: E402

CFG = ChunkConfig(
    max_seconds=30, trigger_seconds=36, overlap_seconds=2, search_seconds=6, max_audio_seconds=300,
    max_audio_bytes=64 << 20, engine_max_file_bytes=180 << 20,
)


def _wav_bytes(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    pcm = np.clip(np.rint(audio * 32767), -32768, 32767).astype("<i2")
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _samples(wav: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(wav)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32767


class FakeEngine:
    def __init__(self, length_on_chunk: int | None = None, fail_on_chunk: int | None = None, hallucinate=None):
        #: ``hallucinate(index, words, duration_s, body)`` -> ``(text, finish_reason)`` or None:
        #: what a runaway engine answers instead of the words (tests/test_runaway.py).
        self.hallucinate = hallucinate
        self.requests: list[dict] = []
        self.raw: list[bytes] = []
        self.length_on_chunk = length_on_chunk
        self.fail_on_chunk = fail_on_chunk
        self.served: dict[str, bytes] = {}
        self.usages: list[dict] = []
        self.wavs: list[bytes] = []
        #: Drop the connection on the next N audio GETs (mid-body) / chat POSTs (unanswered).
        self.drop_audio = 0
        self.drop_chat = 0

    def app(self) -> web.Application:
        app = web.Application(client_max_size=256 << 20)
        app.router.add_post("/v1/chat/completions", self.chat)
        app.router.add_get("/health", self.health)
        app.router.add_get("/audio/{name}", self.audio)
        return app

    async def health(self, _request: web.Request) -> web.Response:
        return web.Response(text="ok")

    async def audio(self, request: web.Request) -> web.Response:
        if self.drop_audio > 0:
            # Cut the transfer mid-body: aiohttp's client retries a GET dropped before the
            # response headers by itself, but not one dropped while the body streams.
            self.drop_audio -= 1
            data = self.served[request.match_info["name"]]
            resp = web.StreamResponse(headers={"Content-Length": str(len(data))})
            await resp.prepare(request)
            await resp.write(data[:100])
            request.transport.close()
            return resp
        return web.Response(body=self.served[request.match_info["name"]])

    async def chat(self, request: web.Request) -> web.Response:
        raw = await request.read()
        if self.drop_chat > 0:
            self.drop_chat -= 1
            request.transport.close()
            return web.Response()
        self.raw.append(raw)
        body = json.loads(raw)
        self.requests.append(body)
        part = next(p for m in body["messages"] for p in m["content"] if p.get("type") == "audio_url")
        url = part["audio_url"]["url"]
        if url.startswith("data:"):
            wav = base64.b64decode(url.partition(",")[2])
        else:
            assert url.startswith("file://"), url
            path = Path(unquote(urlparse(url).path))
            assert path.exists(), "the proxy removed the file before the engine read it"
            wav = path.read_bytes()
        self.wavs.append(wav)
        idx = len(self.requests) - 1
        if self.fail_on_chunk == idx:
            return web.json_response({"error": {"message": "boom", "code": 400}}, status=400)
        samples = _samples(wav)
        words = read_words(samples)
        prompt = int(len(samples) / 16000 * 12.5) + 25
        finish = "length" if self.length_on_chunk == idx else "stop"
        text = " ".join(words)
        if self.hallucinate is not None:
            got = self.hallucinate(idx, words, len(samples) / 16000, body)
            if got is not None:
                text, finish = got
        n_out = len(text.split())
        usage = {"prompt_tokens": prompt, "completion_tokens": n_out, "total_tokens": prompt + n_out}
        self.usages.append(usage)
        return web.json_response({
            "id": f"chatcmpl-{idx}", "object": "chat.completion", "created": 1, "model": "nemotron",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text, "reasoning": None},
                         "finish_reason": finish, "logprobs": None}],
            "usage": usage,
        })


def _chat(url: str, **extra) -> dict:
    return {
        "model": "nemotron",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Transcribe."},
            {"type": "audio_url", "audio_url": {"url": url}},
        ]}],
        "max_tokens": 1000, "temperature": 0, **extra,
    }


@pytest.fixture
async def rig(tmp_path):
    async def make(engine: FakeEngine, cfg: ChunkConfig = CFG, guard: GuardConfig | None = None, retry=False):
        eng_server = TestServer(engine.app())
        await eng_server.start_server()
        upstream = str(eng_server.make_url("")).rstrip("/")
        proxy = chunk_proxy.ChunkProxy(
            cfg, upstream, str(tmp_path / "work"), concurrent.futures.ThreadPoolExecutor(2), guard=guard,
            retry_transient=retry,
        )
        client = TestClient(TestServer(proxy.app()))
        await client.start_server()
        made.append((client, eng_server))
        return client, upstream

    made: list = []
    yield make
    for client, eng in made:
        await client.close()
        await eng.close()


def _expected(n: int) -> str:
    return " ".join(f"w{k}" for k in range(n))


async def test_long_inline_audio_is_chunked_and_stitched(rig, tmp_path):
    audio, _ = word_audio(n_words=90, pause_after={27: 1.5, 55: 1.5})
    engine = FakeEngine()
    client, _ = await rig(engine)
    data = "data:audio/wav;base64," + base64.b64encode(_wav_bytes(audio)).decode()
    r = await client.post("/v1/chat/completions", json=_chat(data))
    assert r.status == 200, await r.text()
    body = await r.json()
    n_chunks = int(r.headers["X-Audio-Chunks"])
    assert n_chunks == len(engine.requests) >= 3
    assert body["choices"][0]["message"]["content"] == _expected(90)
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["object"] == "chat.completion"
    assert body["usage"]["prompt_tokens"] > int(len(audio) / 16000 * 12.5)  # overlaps count twice
    # Every chunk request: non-streaming, file:// audio, the client's other fields kept.
    # max_tokens: the client's 1000, capped by each chunk's duration budget (runaway guard).
    for req, wav in zip(engine.requests, engine.wavs):
        chunk_s = (len(wav) - 44) / 2 / 16000
        assert req["stream"] is False and req["max_tokens"] == min(1000, token_budget(chunk_s, GuardConfig()))
        assert req["messages"][0]["content"][0]["text"] == "Transcribe."
    # Nothing left on disk.
    assert list((tmp_path / "work").iterdir()) == []


async def test_usage_is_the_sum_of_chunk_usage(rig):
    audio, _ = word_audio(n_words=70)
    engine = FakeEngine()
    client, _ = await rig(engine)
    data = "data:audio/wav;base64," + base64.b64encode(_wav_bytes(audio)).decode()
    body = await (await client.post("/v1/chat/completions", json=_chat(data))).json()
    assert len(engine.usages) >= 3
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        assert body["usage"][key] == sum(u[key] for u in engine.usages)
    # The overlap's words were transcribed twice but appear once.
    assert body["usage"]["completion_tokens"] > 70
    assert body["choices"][0]["message"]["content"] == _expected(70)


async def test_length_in_any_chunk_is_reported(rig):
    audio, _ = word_audio(n_words=70)
    client, _ = await rig(FakeEngine(length_on_chunk=1))
    data = "data:audio/wav;base64," + base64.b64encode(_wav_bytes(audio)).decode()
    body = await (await client.post("/v1/chat/completions", json=_chat(data))).json()
    assert body["choices"][0]["finish_reason"] == "length"


async def test_logs_seams_and_capped_chunk_text(rig, caplog):
    audio, _ = word_audio(n_words=70)
    client, _ = await rig(FakeEngine(length_on_chunk=1))
    data = "data:audio/wav;base64," + base64.b64encode(_wav_bytes(audio)).decode()
    with caplog.at_level("DEBUG", logger="chunk_proxy"):
        # max_tokens under the duration budget: the length is the client's, not a runaway.
        r = await client.post("/v1/chat/completions", json=_chat(data, max_tokens=100))
    assert "X-Transcript-Guard" not in r.headers
    n = int(r.headers["X-Audio-Chunks"])
    msgs = [rec.getMessage() for rec in caplog.records]
    seam_lines = [m for m in msgs if " seam " in m and "matched=" in m]
    assert len(seam_lines) == n - 1
    assert all("dropped_prev=" in m and "dropped_next=" in m for m in seam_lines)
    capped = [m for m in msgs if "finish=length" in m and "chars=" in m]
    assert len(capped) == 1 and "chunk 1 " in capped[0]
    assert "head=" not in capped[0] and "tail=" not in capped[0]  # ZDR: no transcript text


async def test_each_chunk_request_carries_exactly_its_span(rig):
    audio, _ = word_audio(n_words=90, pause_after={27: 1.5, 55: 1.5})
    engine = FakeEngine()
    client, _ = await rig(engine)
    wav = _wav_bytes(audio)
    data = "data:audio/wav;base64," + base64.b64encode(wav).decode()
    r = await client.post("/v1/chat/completions", json=_chat(data))
    assert r.status == 200
    with wave.open(io.BytesIO(wav)) as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    frame = int(0.02 * 16000)
    x = pcm.astype(np.float32) / 32768
    energy = (x[: len(x) // frame * frame].reshape(-1, frame).astype(np.float64) ** 2).mean(axis=1)
    from long_audio import plan_chunks

    spans = plan_chunks(energy, len(pcm), CFG)
    assert len(engine.wavs) == len(spans) >= 3
    got = sorted((np.frombuffer(b[44:], dtype="<i2") for b in engine.wavs), key=len)
    want = sorted((pcm[s:e] for s, e in spans), key=len)
    for g, w_ in zip(got, want):
        assert np.array_equal(g, w_)
    # Every chunk carries the client's prompt unchanged.
    assert all(req["messages"][0]["content"][0] == {"type": "text", "text": "Transcribe."} for req in engine.requests)


async def test_chunk_error_is_returned_as_is(rig):
    audio, _ = word_audio(n_words=70)
    client, _ = await rig(FakeEngine(fail_on_chunk=0))
    data = "data:audio/wav;base64," + base64.b64encode(_wav_bytes(audio)).decode()
    r = await client.post("/v1/chat/completions", json=_chat(data))
    assert r.status == 400 and (await r.json())["error"]["message"] == "boom"


async def test_short_inline_audio_passes_through_byte_for_byte_without_guard(rig):
    audio, _ = word_audio(n_words=10)
    engine = FakeEngine()
    client, _ = await rig(engine, guard=GuardConfig(enabled=False))
    raw = json.dumps(_chat("data:audio/wav;base64," + base64.b64encode(_wav_bytes(audio)).decode())).encode()
    r = await client.post("/v1/chat/completions", data=raw, headers={"Content-Type": "application/json"})
    assert r.status == 200 and "X-Audio-Chunks" not in r.headers
    assert engine.raw == [raw]
    assert (await r.json())["choices"][0]["message"]["content"] == _expected(10)


async def test_url_audio_is_fetched_once_and_chunked_when_long(rig):
    engine = FakeEngine()
    client, upstream = await rig(engine)
    short, _ = word_audio(n_words=10)
    long_, _ = word_audio(n_words=80, pause_after={26: 1.0, 52: 1.0})
    engine.served = {"short.wav": _wav_bytes(short), "long.wav": _wav_bytes(long_)}
    r = await client.post("/v1/chat/completions", json=_chat(f"{upstream}/audio/short.wav"))
    assert r.status == 200 and "X-Audio-Chunks" not in r.headers
    assert engine.requests[0]["messages"][0]["content"][1]["audio_url"]["url"].startswith("file://")
    r = await client.post("/v1/chat/completions", json=_chat(f"{upstream}/audio/long.wav"))
    assert r.status == 200 and int(r.headers["X-Audio-Chunks"]) >= 3
    assert (await r.json())["choices"][0]["message"]["content"] == _expected(80)


async def test_stream_true_gets_sse_with_usage(rig):
    audio, _ = word_audio(n_words=70)
    client, _ = await rig(FakeEngine())
    data = "data:audio/wav;base64," + base64.b64encode(_wav_bytes(audio)).decode()
    r = await client.post("/v1/chat/completions", json=_chat(data, stream=True, stream_options={"include_usage": True}))
    assert r.status == 200 and r.headers["Content-Type"].startswith("text/event-stream")
    events = [line[6:] for line in (await r.text()).split("\n\n") if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    objs = [json.loads(e) for e in events[:-1]]
    assert objs[0]["choices"][0]["delta"]["content"] == _expected(70)
    assert objs[1]["choices"][0]["finish_reason"] == "stop"
    assert objs[2]["usage"]["completion_tokens"] >= 70


async def test_refusals(rig, tmp_path):
    client, _ = await rig(FakeEngine())
    r = await client.post("/v1/chat/completions", json=_chat("file:///etc/passwd"))
    assert r.status == 400 and "file://" in (await r.json())["error"]["message"]
    too_long, _ = word_audio(n_words=320)  # ~320 s > max_audio_seconds 300
    data = "data:audio/wav;base64," + base64.b64encode(_wav_bytes(too_long)).decode()
    r = await client.post("/v1/chat/completions", json=_chat(data))
    assert r.status == 400 and "NEMOTRON_MAX_AUDIO_SECONDS" in (await r.json())["error"]["message"]
    audio, _ = word_audio(n_words=70)
    data = "data:audio/wav;base64," + base64.b64encode(_wav_bytes(audio)).decode()
    r = await client.post("/v1/chat/completions", json=_chat(data, n=2))
    assert r.status == 400
    r = await client.post("/v1/chat/completions", json=_chat("data:audio/wav;base64," + "!" * 40000))
    assert r.status == 400


async def test_disabled_and_other_routes_pass_through(rig):
    engine = FakeEngine()
    client, _ = await rig(engine, ChunkConfig(max_seconds=0), guard=GuardConfig(enabled=False))
    assert (await client.get("/health")).status == 200
    audio, _ = word_audio(n_words=70)
    raw = json.dumps(_chat("data:audio/wav;base64," + base64.b64encode(_wav_bytes(audio)).decode())).encode()
    r = await client.post("/v1/chat/completions", data=raw, headers={"Content-Type": "application/json"})
    assert r.status == 200 and engine.raw == [raw]


async def test_guard_without_chunking_is_one_capped_unit(rig):
    # RTX PRO 6000 port: chunking off, guard on. One engine request for the whole file,
    # as a file:// URL, with max_tokens capped by the audio's duration budget.
    engine = FakeEngine()
    client, _ = await rig(engine, ChunkConfig(max_seconds=0))
    audio, _ = word_audio(n_words=70)
    data = "data:audio/wav;base64," + base64.b64encode(_wav_bytes(audio)).decode()
    r = await client.post("/v1/chat/completions", json=_chat(data))
    assert r.status == 200
    assert (await r.json())["choices"][0]["message"]["content"] == _expected(70)
    assert len(engine.requests) == 1
    req = engine.requests[0]
    assert req["max_tokens"] == min(1000, token_budget(len(audio) / 16000, GuardConfig()))
    assert "X-Audio-Chunks" not in r.headers or r.headers["X-Audio-Chunks"] == "1"


async def test_engine_down_is_503(tmp_path):
    proxy = chunk_proxy.ChunkProxy(CFG, "http://127.0.0.1:9", str(tmp_path / "w"), concurrent.futures.ThreadPoolExecutor(1))
    async with TestClient(TestServer(proxy.app())) as client:
        assert (await client.get("/health")).status == 503


def test_build_validates_env():
    with pytest.raises(ValueError):
        chunk_proxy.build({"NEMOTRON_CHUNK_TRIGGER_SECONDS": "4000", "VLLM_MAX_AUDIO_DECODE_DURATION_S": "3600"})
    with pytest.raises(ValueError):
        chunk_proxy.build({"NEMOTRON_CHUNK_DECODE_WORKERS": "0", "VLLM_MAX_AUDIO_DECODE_DURATION_S": "3600", "VLLM_MAX_AUDIO_CLIP_FILESIZE_MB": "180"})
    cfg, opts = chunk_proxy.build({"VLLM_MAX_AUDIO_DECODE_DURATION_S": "3600", "VLLM_MAX_AUDIO_CLIP_FILESIZE_MB": "180"})
    assert (cfg.trigger_seconds, cfg.max_seconds, opts["workers"]) == (3600, 3300, 2)


def test_wav_helper_roundtrip(tmp_path):
    audio, _ = word_audio(n_words=3)
    write_wav(tmp_path / "a.wav", audio)
    assert read_words(_samples((tmp_path / "a.wav").read_bytes())) == ["w0", "w1", "w2"]


async def test_audio_between_chunk_max_and_trigger_is_one_request(rig):
    """Like 55-60 min under the defaults: longer than a chunk, not over the trigger."""
    audio, _ = word_audio(n_words=33)  # ~33 s: over max_seconds 30, under trigger 36
    engine = FakeEngine()
    client, _ = await rig(engine)
    raw = json.dumps(_chat("data:audio/wav;base64," + base64.b64encode(_wav_bytes(audio)).decode())).encode()
    r = await client.post("/v1/chat/completions", data=raw, headers={"Content-Type": "application/json"})
    assert r.status == 200 and "X-Audio-Chunks" not in r.headers and len(engine.requests) == 1
    assert (await r.json())["choices"][0]["message"]["content"] == _expected(33)
    # Without the guard, the request goes to the engine byte for byte.
    engine2 = FakeEngine()
    client2, _ = await rig(engine2, guard=GuardConfig(enabled=False))
    r = await client2.post("/v1/chat/completions", data=raw, headers={"Content-Type": "application/json"})
    assert r.status == 200 and "X-Audio-Chunks" not in r.headers and engine2.raw == [raw]


async def test_failed_requests_are_logged_without_their_body(rig, caplog):
    client, _ = await rig(FakeEngine())
    with caplog.at_level("DEBUG", logger="chunk_proxy"):
        r = await client.post("/v1/chat/completions", json=_chat("file:///etc/passwd"))
    assert r.status == 400
    assert "request_error chat /v1/chat/completions status=400 body_bytes=" in caplog.text
    assert "file://" not in caplog.text  # ZDR: the error body can echo request content


async def test_dropped_audio_fetch_is_a_400_without_retry(rig):
    engine = FakeEngine()
    client, upstream = await rig(engine)
    engine.served = {"a.wav": _wav_bytes(word_audio(n_words=5)[0])}
    engine.drop_audio = 1
    r = await client.post("/v1/chat/completions", json=_chat(f"{upstream}/audio/a.wav"))
    assert r.status == 400
    assert "Could not fetch the audio URL" in (await r.json())["error"]["message"]


async def test_retry_refetches_a_dropped_audio_url_once(rig, caplog):
    engine = FakeEngine()
    client, upstream = await rig(engine, retry=True)
    engine.served = {"a.wav": _wav_bytes(word_audio(n_words=5)[0])}
    engine.drop_audio = 1
    with caplog.at_level("DEBUG", logger="chunk_proxy"):
        r = await client.post("/v1/chat/completions", json=_chat(f"{upstream}/audio/a.wav"))
    assert r.status == 200, await r.text()
    assert (await r.json())["choices"][0]["message"]["content"] == _expected(5)
    assert "transient_retry fetch attempt=1" in caplog.text
    # Dropped twice: the fetch fails, and that is the server's problem, not the client's.
    engine.drop_audio = 2
    r = await client.post("/v1/chat/completions", json=_chat(f"{upstream}/audio/a.wav"))
    assert r.status == 502
    body = await r.json()
    assert body["error"]["type"] == "InternalServerError"
    assert "Could not fetch the audio URL" in body["error"]["message"]


async def test_retry_resends_a_chunk_after_a_dropped_engine_connection(rig, caplog):
    engine = FakeEngine()
    client, _ = await rig(engine, retry=True)
    data = "data:audio/wav;base64," + base64.b64encode(_wav_bytes(word_audio(n_words=70)[0])).decode()
    engine.drop_chat = 1
    with caplog.at_level("DEBUG", logger="chunk_proxy"):
        r = await client.post("/v1/chat/completions", json=_chat(data))
    assert r.status == 200, await r.text()
    assert (await r.json())["choices"][0]["message"]["content"] == _expected(70)
    assert "transient_retry engine attempt=1" in caplog.text


async def test_dropped_engine_connection_is_a_502_without_retry(rig):
    engine = FakeEngine()
    client, _ = await rig(engine)
    data = "data:audio/wav;base64," + base64.b64encode(_wav_bytes(word_audio(n_words=70)[0])).decode()
    engine.drop_chat = 1
    r = await client.post("/v1/chat/completions", json=_chat(data))
    assert r.status == 502
    assert "engine request for a chunk failed" in (await r.json())["error"]["message"]


def test_proxy_retry_flag_parses():
    env = {"VLLM_MAX_AUDIO_DECODE_DURATION_S": "3600", "VLLM_MAX_AUDIO_CLIP_FILESIZE_MB": "180"}
    assert chunk_proxy.build(env)[1]["retry_transient"] is False
    assert chunk_proxy.build({**env, "NEMOTRON_PROXY_RETRY": "1"})[1]["retry_transient"] is True
    with pytest.raises(ValueError, match="NEMOTRON_PROXY_RETRY"):
        chunk_proxy.build({**env, "NEMOTRON_PROXY_RETRY": "2"})


async def test_retry_keeps_pooled_connections_under_the_engine_keepalive(rig):
    for retry, want in ((False, 15.0), (True, chunk_proxy._KEEPALIVE_S)):
        client, _ = await rig(FakeEngine(), retry=retry)
        proxy = next(cb.__self__ for cb in client.server.app.on_startup if getattr(cb, "__self__", None))
        for session in (proxy.session, proxy.fetch_session):
            assert session.connector._keepalive_timeout == want
    assert chunk_proxy._KEEPALIVE_S < 5  # vLLM's VLLM_HTTP_TIMEOUT_KEEP_ALIVE default


def test_log_level_env():
    import logging

    assert chunk_proxy.log_level({}) == logging.INFO
    assert chunk_proxy.log_level({"NEMOTRON_LOG_LEVEL": "debug"}) == logging.DEBUG
    with pytest.raises(ValueError, match="NEMOTRON_LOG_LEVEL"):
        chunk_proxy.log_level({"NEMOTRON_LOG_LEVEL": "verbose"})


async def test_diagnostics_are_debug_only(rig, caplog):
    """At the default INFO level a failed request and a chunked request log nothing."""
    client, _ = await rig(FakeEngine())
    with caplog.at_level("INFO", logger="chunk_proxy"):
        await client.post("/v1/chat/completions", json=_chat("file:///etc/passwd"))
        audio, _ = word_audio(n_words=70)
        r = await client.post("/v1/chat/completions", json=_chat("data:audio/wav;base64," + base64.b64encode(_wav_bytes(audio)).decode()))
    assert r.status == 200 and int(r.headers["X-Audio-Chunks"]) > 1
    assert [rec for rec in caplog.records if rec.name == "chunk_proxy"] == []
