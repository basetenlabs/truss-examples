"""The sidecar engine: the request it sends, what it makes of the reply, and how it starts.

Run with the stream tests (see test_nemotron_stream.py). The sidecar itself needs the image's
GPU stack, so it is an ``httpx.MockTransport`` here, and startup is driven with real
subprocesses that never become healthy.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import subprocess
import wave

import httpx
import pytest
from baseten_stt.backend.base import TurnDecodeError
from nemotron_stt import engine
from baseten_stt.protocol.capabilities import GrantedContext
from nemotron_stt.engine import (
    PROMPT,
    _SidecarEngine,
    _wait_for_sidecar,
    parse_response,
    pcm_to_wav,
    prompt_for,
    request_body,
)

ONE_SECOND = b"\x01\x00" * engine.SAMPLE_RATE


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url="http://sidecar", transport=httpx.MockTransport(handler))


def test_wav_round_trips_pcm():
    with wave.open(io.BytesIO(pcm_to_wav(ONE_SECOND)), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16_000)
        assert wav.readframes(wav.getnframes()) == ONE_SECOND


def test_request_carries_instruction_audio_and_scaled_cap():
    body = request_body(ONE_SECOND * 10)
    text, audio = body["messages"][0]["content"]
    # Without the instruction the model answers about the audio instead of transcribing it.
    assert text == {"type": "text", "text": PROMPT}
    uri = audio["audio_url"]["url"]
    assert uri.startswith("data:audio/wav;base64,")
    assert base64.b64decode(uri.split(",", 1)[1]) == pcm_to_wav(ONE_SECOND * 10)
    assert body["temperature"] == 0
    assert body["max_tokens"] == 10 * engine.TOKENS_PER_AUDIO_S + 32
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_a_session_that_sets_nothing_keeps_the_measured_prompt():
    assert prompt_for() == prompt_for("auto", None) == PROMPT
    assert prompt_for("auto", GrantedContext()) == PROMPT


def test_a_named_language_takes_the_cards_instruction():
    assert prompt_for("en") == (
        "Audio transcription. Language: English. Wording: verbatim, including fillers and "
        "repetitions. Formatting: punctuation and capitalization. Output: transcript only."
    )
    assert "Language: Tamil." in prompt_for("ta")


def test_context_follows_the_instruction():
    context = GrantedContext(
        text="Transcribe only the speaker nearest to the microphone.",
        keyterms=("Yusuf Hernandez", "W2702727"),
    )
    assert prompt_for("auto", context) == (
        "Transcribe this audio. Transcribe only the speaker nearest to the microphone."
        " Context terms: Yusuf Hernandez, W2702727."
    )
    assert prompt_for("en", GrantedContext(keyterms=("Elm Street",))).endswith(
        "Output: transcript only. Context terms: Elm Street."
    )


def test_the_request_carries_the_prompt_it_is_given():
    text, _ = request_body(ONE_SECOND, "Transcribe this audio. Context terms: X.")["messages"][0][
        "content"
    ]
    assert text == {"type": "text", "text": "Transcribe this audio. Context terms: X."}


@pytest.mark.parametrize(
    ("body", "text"),
    [
        ({"choices": [{"message": {"content": "  hello there \n"}}]}, "hello there"),
        ({"choices": [{"message": {"content": None}}]}, ""),
        ({"choices": [{"message": {"content": ""}}]}, ""),
        ({"choices": []}, ""),
        ({}, ""),
    ],
)
def test_parse_response(body, text):
    assert parse_response(body) == text


def test_a_decode_that_ran_out_of_tokens_fails_the_turn():
    # Truncated text returned as a final would lose the rest of the turn silently.
    body = {"choices": [{"message": {"content": "the the the"}, "finish_reason": "length"}]}
    with pytest.raises(TurnDecodeError):
        parse_response(body)


@pytest.mark.asyncio
async def test_transcribe_posts_to_chat_completions():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "hola mundo"}}]})

    async with _client(handler) as client:
        assert await _SidecarEngine(client).transcribe(ONE_SECOND) == ("hola mundo", None)
    assert [r.url.path for r in seen] == ["/v1/chat/completions"]


@pytest.mark.asyncio
async def test_transcribe_sends_the_sessions_instruction_and_reports_a_named_language():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "hello"}}]})

    context = GrantedContext(keyterms=("Elm Street",))
    async with _client(handler) as client:
        assert await _SidecarEngine(client).transcribe(ONE_SECOND, "en", context) == (
            "hello",
            "en",
        )
    assert seen[0]["messages"][0]["content"][0]["text"] == prompt_for("en", context)


@pytest.mark.asyncio
async def test_transcribe_raises_on_sidecar_error():
    async with _client(lambda r: httpx.Response(400, json={"error": "bad audio"})) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await _SidecarEngine(client).transcribe(ONE_SECOND)


@pytest.mark.asyncio
async def test_cancelling_a_decode_abandons_the_request():
    started = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        started.set()
        await asyncio.sleep(30)
        return httpx.Response(200, json={"choices": [{"message": {"content": "late"}}]})

    async with _client(handler) as client:
        task = asyncio.create_task(_SidecarEngine(client).transcribe(ONE_SECOND))
        await asyncio.wait_for(started.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)


def test_wait_fails_fast_when_the_sidecar_exits(monkeypatch):
    monkeypatch.setattr(engine, "SIDECAR_URL", "http://127.0.0.1:9")  # nothing listens
    proc = subprocess.Popen(["/bin/sh", "-c", "exit 3"])
    with pytest.raises(RuntimeError, match="exited with code 3"):
        _wait_for_sidecar(proc, timeout_s=60)


def test_wait_times_out_and_stops_the_sidecar(monkeypatch):
    monkeypatch.setattr(engine, "SIDECAR_URL", "http://127.0.0.1:9")
    monkeypatch.setattr(engine.time, "sleep", lambda s: None)
    proc = subprocess.Popen(["/bin/sh", "-c", "sleep 60"])
    with pytest.raises(RuntimeError, match="not healthy"):
        _wait_for_sidecar(proc, timeout_s=0.5)
    assert proc.wait(timeout=10) is not None


def test_a_sidecar_exit_stops_the_server(monkeypatch):
    """vLLM dying after startup must end the process, since the probes would stay green."""
    signals = []
    monkeypatch.setattr(engine.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(engine.time, "sleep", lambda s: None)
    monkeypatch.setattr(engine.os, "_exit", lambda code: signals.append(("exit", code)))
    proc = subprocess.Popen(["/bin/sh", "-c", "exit 1"])
    engine._supervise(proc)
    assert signals == [(engine.os.getpid(), engine.signal.SIGTERM), ("exit", 1)]


def test_warmup_audio_is_not_silence():
    pcm = engine._warmup_audio(1.0)
    assert len(pcm) == 2 * engine.SAMPLE_RATE
    assert any(pcm)
