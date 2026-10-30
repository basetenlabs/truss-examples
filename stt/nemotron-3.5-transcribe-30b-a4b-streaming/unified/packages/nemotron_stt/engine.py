"""Running Nemotron 3.5 Transcribe, and the handle the backend drives it through.

:func:`build` is the entry point: baseten-stt's server imports it by the name in
``BASETEN_STT_BACKEND`` and awaits the backend it returns.

Unlike the Qwen3-ASR template, the engine is not loaded in this process. The model needs NeMo
Speech's vLLM plugin on vLLM 0.28 with a CUDA 13 torch stack, which NVIDIA validates on
Python 3.13 -- while baseten-stt runs on the server image's Python 3.12 with its own pins. So
the engine is the batch preset's ``vllm serve``, launched as a localhost sidecar from its own
venv, and a decode is one chat-completions request to it.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import random
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
import wave
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import httpx
from baseten_stt.backend.base import TurnDecodeError

if TYPE_CHECKING:
    from nemotron_stt.backend import NemotronBackend

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
BYTES_PER_SEC = SAMPLE_RATE * 2  # int16 mono
SERVED_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"
SIDECAR_PORT = int(os.environ.get("NEMOTRON_SIDECAR_PORT", "8000"))
SIDECAR_URL = f"http://127.0.0.1:{SIDECAR_PORT}"
SIDECAR_SCRIPT = os.environ.get("NEMOTRON_SIDECAR_SCRIPT", "/packages/sidecar/start_vllm.sh")
#: The sidecar loads the mounted weights and initialises the engine: ~300 s of engine init with the
#: fused-MoE kernel compiled into the image, 25-45 min when FlashInfer had to compile it at start.
#: Under truss's startup threshold.
SIDECAR_STARTUP_TIMEOUT_S = float(os.environ.get("NEMOTRON_SIDECAR_STARTUP_TIMEOUT_S", "2700"))
#: See ``nemotron_stt.DEFAULT_PARTIAL_WINDOW_S``.
PARTIAL_WINDOW_S = float(os.environ.get("STREAM_PARTIAL_WINDOW_S", "8.0"))
#: The model is a chat model: without an instruction it answers questions about the audio
#: instead of transcribing it. This is the prompt b10-bench sends, which scored 1.5% WER on
#: pipecat and kept non-English speech in its own language.
PROMPT = "Transcribe this audio."
#: Speech runs ~3.3 output tokens per second of audio (11.7k for the 1 hr podcast); the cap is
#: generous against that, and bounds a runaway decode.
TOKENS_PER_AUDIO_S = 12
#: Per decode. A final decodes the whole turn; a long monologue is still seconds, not minutes.
REQUEST_TIMEOUT_S = float(os.environ.get("NEMOTRON_REQUEST_TIMEOUT_S", "120"))
#: The first decodes after the sidecar reports healthy JIT-compile Triton kernels, long enough
#: that a session's finals used to time out. Warmup runs them before the backend is published,
#: so readiness covers them. Turn lengths up to the server's 29 s cut, sequentially, then
#: ``predict_concurrency`` of them at once for the batched shapes. 0 skips warmup.
WARMUP = os.environ.get("NEMOTRON_WARMUP", "1") != "0"
WARMUP_AUDIO_S = (1.0, 4.0, 8.0, 16.0, 29.0)
WARMUP_CONCURRENCY = 32
#: Per warmup decode: the first carries the kernel compiles, not just the audio.
WARMUP_REQUEST_TIMEOUT_S = float(os.environ.get("NEMOTRON_WARMUP_REQUEST_TIMEOUT_S", "900"))
#: How long a server whose sidecar died gets to shut down on its own before it is exited.
SIDECAR_EXIT_GRACE_S = 30.0


def pcm_to_wav(pcm: bytes) -> bytes:
    """Wrap int16 mono PCM in a WAV header, which is what the chat route's audio loader reads."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm)
    return buf.getvalue()


def request_body(pcm: bytes) -> dict:
    """The chat-completions request for one decode of ``pcm``."""
    uri = "data:audio/wav;base64," + base64.b64encode(pcm_to_wav(pcm)).decode("ascii")
    return {
        "model": SERVED_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "audio_url", "audio_url": {"url": uri}},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": int(len(pcm) / BYTES_PER_SEC * TOKENS_PER_AUDIO_S) + 32,
        # Already the sidecar's default; stated so the request does not depend on it.
        "chat_template_kwargs": {"enable_thinking": False},
    }


def parse_response(body: dict) -> str:
    """The transcript from a chat-completions response; empty when the model returned none.

    A decode that ran out of ``max_tokens`` is a truncated transcript -- the budget is ~4x
    the model's rate, so in practice a runaway repetition -- and is refused rather than
    returned as a final: :class:`TurnDecodeError` fails this turn and leaves the session up.
    """
    choices = body.get("choices") or []
    if not choices:
        return ""
    if choices[0].get("finish_reason") == "length":
        raise TurnDecodeError("transcript exceeded the decode budget for this turn's audio")
    return (choices[0].get("message", {}).get("content") or "").strip()


class _SidecarEngine:
    """The model handle the backend holds: one decode, awaited."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def transcribe(self, pcm: bytes, language: str = "auto") -> tuple[str, str | None]:
        """One full decode of ``pcm``. The language is never forced: the model auto-detects
        and reports none, so the second element is always None.

        Cancelling the task awaiting this closes the request's connection, and vLLM aborts a
        request whose client went away -- so a cleared turn frees the GPU.
        """
        resp = await self._client.post("/v1/chat/completions", json=request_body(pcm))
        resp.raise_for_status()
        return parse_response(resp.json()), None


def _wait_for_sidecar(proc: subprocess.Popen, timeout_s: float) -> None:
    """Block until the sidecar answers /health, failing fast if it exits first."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        code = proc.poll()
        if code is not None:
            raise RuntimeError(f"engine sidecar exited with code {code} before becoming healthy")
        try:
            with urllib.request.urlopen(f"{SIDECAR_URL}/health", timeout=5) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(5)
    proc.terminate()
    raise RuntimeError(f"engine sidecar not healthy after {timeout_s:.0f}s")


def _warmup_audio(seconds: float) -> bytes:
    """Seeded noise at speech level: silence decodes to nothing, which would leave the decode
    path cold."""
    rng = random.Random(0)
    n = int(seconds * SAMPLE_RATE)
    return b"".join(rng.randint(-3000, 3000).to_bytes(2, "little", signed=True) for _ in range(n))


def _warmup() -> None:
    """Decode representative turns before the backend is published, failing startup if the
    engine cannot. The transcripts are discarded; only completing matters."""
    clips = [_warmup_audio(s) for s in WARMUP_AUDIO_S]
    started = time.monotonic()
    with httpx.Client(
        base_url=SIDECAR_URL,
        timeout=httpx.Timeout(WARMUP_REQUEST_TIMEOUT_S, connect=5.0),
        limits=httpx.Limits(max_connections=WARMUP_CONCURRENCY),
    ) as client:

        def decode(pcm: bytes) -> None:
            client.post("/v1/chat/completions", json=request_body(pcm)).raise_for_status()

        for seconds, pcm in zip(WARMUP_AUDIO_S, clips):
            t0 = time.monotonic()
            decode(pcm)
            logger.info("warmup: %.0fs turn in %.1fs", seconds, time.monotonic() - t0)
        burst = [clips[i % len(clips)] for i in range(WARMUP_CONCURRENCY)]
        t0 = time.monotonic()
        with ThreadPoolExecutor(WARMUP_CONCURRENCY) as pool:
            list(pool.map(decode, burst))
        logger.info("warmup: %d concurrent turns in %.1fs", len(burst), time.monotonic() - t0)
    logger.info("warmup done in %.1fs", time.monotonic() - started)


def _supervise(proc: subprocess.Popen) -> None:
    """Take the server down with the sidecar.

    baseten-stt's probes only say the backend was built, so a replica whose vLLM died would
    stay ready and fail every decode. SIGTERM lets uvicorn shut down; the hard exit covers a
    shutdown that stalls on open sessions. Either way the process ends and the platform
    replaces the replica.
    """
    code = proc.wait()
    logger.error("engine sidecar exited with code %s; stopping the server", code)
    os.kill(os.getpid(), signal.SIGTERM)
    time.sleep(SIDECAR_EXIT_GRACE_S)
    logger.error("server still up %.0fs after the sidecar exited; exiting", SIDECAR_EXIT_GRACE_S)
    os._exit(1)


def build() -> NemotronBackend:
    """The factory ``BASETEN_STT_BACKEND=nemotron_stt:build`` resolves to.

    Synchronous on purpose, like the template's: bringing the engine up blocks for minutes,
    and a coroutine would only move that block onto the event loop about to serve sessions.
    The sidecar inherits this process's stdout and stderr, so its download and vLLM logs land
    in the deployment logs beside the server's.
    """
    from nemotron_stt.backend import NemotronBackend

    logger.info("starting engine sidecar: %s", SIDECAR_SCRIPT)
    proc = subprocess.Popen(["/bin/bash", SIDECAR_SCRIPT])
    _wait_for_sidecar(proc, SIDECAR_STARTUP_TIMEOUT_S)
    if WARMUP:
        try:
            _warmup()
        except Exception:
            proc.terminate()
            raise
    threading.Thread(target=_supervise, args=(proc,), name="sidecar-supervisor", daemon=True).start()
    client = httpx.AsyncClient(
        base_url=SIDECAR_URL,
        timeout=httpx.Timeout(REQUEST_TIMEOUT_S, connect=5.0),
        # One connection per concurrent decode, finals and partials alike; vLLM batches them.
        limits=httpx.Limits(max_connections=512, max_keepalive_connections=128),
    )
    logger.info("Nemotron 3.5 Transcribe sidecar ready, serving the baseten-stt protocol")
    return NemotronBackend(_SidecarEngine(client), partial_window_s=PARTIAL_WINDOW_S)
