"""HTTP front for vLLM: transcribes long audio in chunks and guards every transcript.

start.sh runs vLLM on 127.0.0.1:8001 and this proxy on :8000. Every request passes
through to vLLM untouched, except ``POST /v1/chat/completions`` with one audio part:

    request ──▶ one audio part? ──no─────────────────────────────▶ vLLM, body unchanged
                   │ yes: inline audio written here, URL audio downloaded once
                   ▼ (vLLM then reads the file, no 2nd fetch)
              metadata duration ≤ NEMOTRON_CHUNK_TRIGGER_SECONDS and file within the
              engine's size limit ──yes──▶ one engine request, audio as file://
                   │ no (or unknown)
                   ▼
              decode once in a worker process → chunk WAVs (long_audio.decode_and_split)
              → all chunks to vLLM at once (stream off) → stitch → one completion

Runaway guard (runaway.py), on both paths unless NEMOTRON_RUNAWAY_GUARD=0 or the request
turns thinking on: each engine request's max_tokens is capped by its audio duration;
a transcript that hit that cap, runs faster than speech or loops is retried once as two
halves (split at a pause), and one still anomalous gets NVIDIA's loop cleanup and
finish_reason "length". Header X-Transcript-Guard reports what happened.

``/v1/audio/transcriptions`` is not served: vLLM registers it only for models that
declare transcription support, and NeMo's nemo_speechlm plugin (a837f4e) does not.

NEMOTRON_PROXY_RETRY=1 (opt-in, default 0) retries the audio-URL fetch and each engine POST
once on a dropped connection, answers a failed fetch with 502 instead of 400 and keeps pooled
connections under vLLM's 5 s keep-alive. Every failed request logs one DEBUG "request_error"
line with its status and body size.

NEMOTRON_LOG_LEVEL (DEBUG | INFO | WARNING | ERROR; default INFO) sets the proxy's log level. At
INFO the proxy logs only its startup config line; every per-request diagnostic (request_error,
transient_retry, chunked-audio summaries and seams, runaway-guard units and retries) is DEBUG,
so NEMOTRON_LOG_LEVEL=DEBUG turns them back on. None of them log request or transcript content
(ZDR): only sizes, counts, timings and error types.

Run ``python3 chunk_proxy.py --check`` to validate the configuration without serving.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import concurrent.futures
import copy
import json
import logging
import multiprocessing
import os
import shutil
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp import web

sys.path.insert(0, str(Path(__file__).resolve().parent))
import long_audio  # noqa: E402
import runaway  # noqa: E402
from long_audio import AudioInputError, Chunk, ChunkConfig  # noqa: E402
from runaway import GuardConfig  # noqa: E402

log = logging.getLogger("chunk_proxy")

LOG_LEVEL_ENV = "NEMOTRON_LOG_LEVEL"
_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


def log_level(env=os.environ) -> int:
    """The proxy logger's level from NEMOTRON_LOG_LEVEL (default INFO)."""
    name = (env.get(LOG_LEVEL_ENV) or "INFO").strip().upper()
    if name not in _LOG_LEVELS:
        raise ValueError(f"{LOG_LEVEL_ENV} must be one of {', '.join(_LOG_LEVELS)}, got {env.get(LOG_LEVEL_ENV)!r}")
    return getattr(logging, name)

CHAT_PATH = "/v1/chat/completions"
_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te",
    "trailers", "transfer-encoding", "upgrade", "content-length", "host",
}
# Bodies above this are parsed off the event loop (a 150 MB data URI takes ~1 s).
_INLINE_PARSE_BYTES = 1 << 20


def _fwd_headers(headers) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in _HOP}


_ERROR_LOG_BYTES = 600


def _log_error(where: str, path: str, status: int, body: bytes) -> None:
    """One line per failed request. Logs the error body's size, never its text: an engine
    error can echo request content, which ZDR keeps out of logs."""
    log.debug("request_error %s %s status=%d body_bytes=%d", where, path, status, len(body))


# Connection-level failures where the peer dropped a pooled keep-alive connection (or the
# transfer) under us; one fresh attempt is the usual remedy (NEMOTRON_PROXY_RETRY).
_TRANSIENT = (aiohttp.ServerDisconnectedError, aiohttp.ClientOSError, aiohttp.ClientPayloadError)
# With NEMOTRON_PROXY_RETRY=1, pooled connections idle at most this long. vLLM's uvicorn closes
# an idle keep-alive connection after 5 s (VLLM_HTTP_TIMEOUT_KEEP_ALIVE) and aiohttp keeps one
# for 15 s by default, so the proxy reused connections the engine had already closed.
_KEEPALIVE_S = 4.0


class AudioFetchError(AudioInputError):
    """The audio URL could not be fetched for a transport reason: not the client's fault."""

    status = 502


def _error(status: int, message: str) -> web.Response:
    kind = "BadRequestError" if status == 400 else "InternalServerError"
    body = {"error": {"message": message, "type": kind, "param": None, "code": status}}
    return web.json_response(body, status=status)


def find_audio(body: dict) -> list[tuple[dict, str]]:
    """``(content part, kind)`` of every audio part; kind is ``audio_url`` or ``input_audio``."""
    found = []
    for msg in body.get("messages") or []:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("audio_url", "input_audio"):
                found.append((part, part["type"]))
    return found


def _audio_url(part: dict) -> str:
    v = part.get("audio_url")
    return v if isinstance(v, str) else (v or {}).get("url", "")


def _set_audio_file(part: dict, kind: str, path: str) -> None:
    url = Path(path).resolve().as_uri()
    if kind == "input_audio":
        part.clear()
        part.update({"type": "audio_url", "audio_url": {"url": url}})
    elif isinstance(part.get("audio_url"), dict):
        part["audio_url"]["url"] = url
    else:
        part["audio_url"] = {"url": url}


def _inline_b64(part: dict, kind: str) -> str | None:
    if kind == "input_audio":
        return (part.get("input_audio") or {}).get("data") or ""
    url = _audio_url(part)
    if url[:5].lower() == "data:":
        return url.partition(",")[2]
    return None


def _content(resp: dict) -> str:
    return (resp["choices"][0].get("message") or {}).get("content") or ""


@dataclass
class GuardReport:
    """What the runaway guard did for one request (response header X-Transcript-Guard)."""

    #: Units (the whole file, or chunks) whose first transcript looked like a runaway.
    flagged: int = 0
    #: Of those, retried as two halves.
    retried: int = 0
    #: Retries whose two halves were both clean.
    fixed: int = 0
    #: Units still anomalous after the retry (finish_reason is set to "length").
    still_anomalous: int = 0
    #: Of those, changed by NVIDIA's loop cleanup.
    cleaned: int = 0

    def header(self) -> str:
        return (
            f"flagged={self.flagged}; retried={self.retried}; fixed={self.fixed}; "
            f"anomalous={self.still_anomalous}; cleaned={self.cleaned}"
        )


class ChunkProxy:
    def __init__(
        self,
        cfg: ChunkConfig,
        upstream: str,
        work_dir: str,
        executor: concurrent.futures.Executor,
        fetch_timeout_s: float = 300.0,
        guard: GuardConfig | None = None,
        retry_transient: bool = False,
    ):
        self.cfg = cfg
        #: Retry the audio fetch and each engine POST once on a dropped connection, answer a
        #: failed fetch with 502 instead of 400, and keep idle pooled connections under vLLM's
        #: 5 s keep-alive (NEMOTRON_PROXY_RETRY=1).
        self.retry_transient = retry_transient
        self.guard = guard if guard is not None else GuardConfig()
        self.upstream = upstream.rstrip("/")
        self.work_dir = Path(work_dir)
        self.executor = executor
        self.fetch_timeout_s = fetch_timeout_s
        self.session: aiohttp.ClientSession | None = None
        self.fetch_session: aiohttp.ClientSession | None = None

    # --- app wiring -------------------------------------------------------------------

    def app(self) -> web.Application:
        # Room for the largest inline audio (base64 is 4/3 the size) plus the JSON around it.
        app = web.Application(client_max_size=self.cfg.max_audio_bytes * 4 // 3 + (16 << 20))
        app.router.add_route("POST", CHAT_PATH, self.chat)
        app.router.add_route("*", "/{tail:.*}", self.passthrough)
        app.on_startup.append(self._start)
        app.on_cleanup.append(self._stop)
        return app

    async def _start(self, _app) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        for leftover in self.work_dir.iterdir():
            shutil.rmtree(leftover, ignore_errors=True)
        no_timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=None)
        keepalive = {"keepalive_timeout": _KEEPALIVE_S} if self.retry_transient else {}
        self.session = aiohttp.ClientSession(
            timeout=no_timeout, auto_decompress=False,
            connector=aiohttp.TCPConnector(limit=0, force_close=False, **keepalive),
        )
        self.fetch_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.fetch_timeout_s), connector=aiohttp.TCPConnector(limit=0, **keepalive)
        )

    async def _stop(self, _app) -> None:
        for s in (self.session, self.fetch_session):
            if s is not None:
                await s.close()

    # --- pass-through -----------------------------------------------------------------

    async def passthrough(self, request: web.Request, body: bytes | None = None) -> web.StreamResponse:
        assert self.session is not None
        data = body if body is not None else (request.content if request.body_exists else None)
        try:
            async with self.session.request(
                request.method, self.upstream + request.path_qs,
                headers=_fwd_headers(request.headers), data=data, allow_redirects=False,
            ) as up:
                resp = web.StreamResponse(status=up.status, reason=up.reason, headers=_fwd_headers(up.headers))
                await resp.prepare(request)
                head = b""
                async for piece in up.content.iter_any():
                    if up.status >= 400 and len(head) < _ERROR_LOG_BYTES:
                        head += piece[: _ERROR_LOG_BYTES - len(head)]
                    await resp.write(piece)
                await resp.write_eof()
                if up.status >= 400:
                    _log_error("passthrough", request.path, up.status, head)
                return resp
        except aiohttp.ClientConnectionError as e:
            # vLLM still starting (or gone): what the readiness probe should see.
            return _error(503, f"engine unavailable: {e}")

    # --- chat completions -------------------------------------------------------------

    async def chat(self, request: web.Request) -> web.StreamResponse:
        resp = await self._chat(request)
        if resp.status >= 400 and isinstance(resp, web.Response) and resp.body is not None:
            _log_error("chat", request.path, resp.status, bytes(resp.body)[:_ERROR_LOG_BYTES])
        return resp

    async def _chat(self, request: web.Request) -> web.StreamResponse:
        raw = await request.read()
        # RTX PRO 6000 batch-precision port: the guard also runs with chunking off
        # (NEMOTRON_CHUNK_MAX_SECONDS=0); start.sh then runs this proxy for the guard alone.
        if not self.cfg.enabled and not self.guard.enabled:
            return await self.passthrough(request, raw)
        loop = asyncio.get_running_loop()
        try:
            body = json.loads(raw) if len(raw) < _INLINE_PARSE_BYTES else await loop.run_in_executor(None, json.loads, raw)
        except ValueError:
            return await self.passthrough(request, raw)  # vLLM reports the bad JSON
        if not isinstance(body, dict):
            return await self.passthrough(request, raw)
        parts = find_audio(body)
        if len(parts) != 1:
            return await self.passthrough(request, raw)  # none, or several (vLLM refuses >1)
        part, kind = parts[0]
        # The runaway guard (runaway.py) covers transcription; with thinking on, the
        # completion is reasoning plus an answer, which a per-second budget does not fit.
        guard = self.guard.enabled and (body.get("chat_template_kwargs") or {}).get("enable_thinking") is not True
        b64 = _inline_b64(part, kind)
        url = None if b64 is not None else _audio_url(part)
        if not guard and b64 is not None and len(b64) * 3 // 4 < self.cfg.inline_probe_min_bytes:
            return await self.passthrough(request, raw)
        if url is not None and not url.lower().startswith(("http://", "https://")):
            if url.lower().startswith("file:"):
                return _error(400, "file:// audio URLs are not accepted; send an http(s) URL or inline data.")
            return await self.passthrough(request, raw)

        req_dir = self.work_dir / uuid.uuid4().hex
        req_dir.mkdir(parents=True)
        try:
            src = req_dir / "source"
            try:
                if b64 is not None:
                    await loop.run_in_executor(None, self._write_b64, b64, src)
                else:
                    await self._download(url, src)
                duration = await loop.run_in_executor(None, long_audio.probe_duration, src)
            except AudioInputError as e:
                return _error(getattr(e, "status", 400), str(e))
            size = src.stat().st_size
            if duration is not None and duration > self.cfg.max_audio_seconds:
                return _error(
                    400,
                    f"Audio is {duration:.0f} s, longer than the {self.cfg.max_audio_seconds:g} s this "
                    "deployment accepts per request (NEMOTRON_MAX_AUDIO_SECONDS).",
                )
            if duration is not None and duration <= self.cfg.trigger_seconds and size <= self.cfg.engine_max_file_bytes:
                simple = (body.get("n") or 1) == 1 and not body.get("logprobs") and not body.get("top_logprobs")
                if guard and simple:
                    unit = Chunk(index=0, start=0, end=max(1, round(duration * long_audio.SAMPLE_RATE)), path=str(src))
                    return await self._transcribe(request, body, kind, [unit], req_dir, duration, guard, None)
                if not guard and b64 is not None:
                    return await self.passthrough(request, raw)
                fwd = copy.deepcopy(body)
                if guard:  # n > 1 or logprobs: the duration cap only
                    runaway.apply_token_cap(fwd, duration, self.guard)
                _set_audio_file(find_audio(fwd)[0][0], kind, str(src))
                return await self.passthrough(request, json.dumps(fwd).encode())
            return await self._chunked(request, body, kind, src, req_dir, duration, guard)
        finally:
            await loop.run_in_executor(None, shutil.rmtree, req_dir, True)

    def _write_b64(self, b64: str, dst: Path) -> None:
        if len(b64) * 3 // 4 > self.cfg.max_audio_bytes:
            raise AudioInputError(
                f"Inline audio is over {self.cfg.max_audio_bytes >> 20} MiB (NEMOTRON_MAX_AUDIO_MB)."
            )
        try:
            dst.write_bytes(base64.b64decode(b64, validate=True))
        except (binascii.Error, ValueError) as e:
            raise AudioInputError(f"Inline audio is not valid base64: {e}") from e

    async def _download(self, url: str, dst: Path) -> None:
        attempts = 2 if self.retry_transient else 1
        err = AudioFetchError if self.retry_transient else AudioInputError
        for attempt in range(1, attempts + 1):
            try:
                return await self._download_once(url, dst)
            except _TRANSIENT as e:
                if attempt < attempts:
                    # The error type only: aiohttp errors can carry the audio URL's host (ZDR).
                    log.debug("transient_retry fetch attempt=%d error=%s", attempt, type(e).__name__)
                    continue
                raise err(f"Could not fetch the audio URL: {e!r}") from e
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                raise err(f"Could not fetch the audio URL: {e!r}") from e

    async def _download_once(self, url: str, dst: Path) -> None:
        assert self.fetch_session is not None
        async with self.fetch_session.get(url) as r:
            if r.status != 200:
                raise AudioInputError(f"Fetching the audio URL returned HTTP {r.status}.")
            if r.content_length and r.content_length > self.cfg.max_audio_bytes:
                raise AudioInputError(f"Audio URL is over {self.cfg.max_audio_bytes >> 20} MiB (NEMOTRON_MAX_AUDIO_MB).")
            got = 0
            with open(dst, "wb") as fh:
                async for piece in r.content.iter_chunked(1 << 20):
                    got += len(piece)
                    if got > self.cfg.max_audio_bytes:
                        raise AudioInputError(
                            f"Audio URL is over {self.cfg.max_audio_bytes >> 20} MiB (NEMOTRON_MAX_AUDIO_MB)."
                        )
                    fh.write(piece)

    async def _chunked(
        self, request: web.Request, body: dict, kind: str, src: Path, req_dir: Path, duration: float | None, guard: bool
    ) -> web.StreamResponse:
        if (body.get("n") or 1) != 1:
            return _error(400, "n > 1 is not supported for audio that is transcribed in chunks.")
        if body.get("logprobs") or body.get("top_logprobs"):
            return _error(400, "logprobs are not supported for audio that is transcribed in chunks.")
        loop = asyncio.get_running_loop()
        t0 = time.monotonic()
        try:
            chunks: list[Chunk] = await loop.run_in_executor(
                self.executor, long_audio.decode_and_split, str(src), str(req_dir), self.cfg
            )
        except AudioInputError as e:
            return _error(400, str(e))
        return await self._transcribe(request, body, kind, chunks, req_dir, duration, guard, time.monotonic() - t0, t0)

    def _unit_body(self, body: dict, kind: str, ch: Chunk, guard: bool) -> tuple[dict, int | None]:
        b = copy.deepcopy(body)
        b["stream"] = False
        b.pop("stream_options", None)
        _set_audio_file(find_audio(b)[0][0], kind, ch.path)
        cap = runaway.apply_token_cap(b, ch.end_s - ch.start_s, self.guard) if guard else None
        return b, cap

    async def _transcribe(
        self,
        request: web.Request,
        body: dict,
        kind: str,
        chunks: list[Chunk],
        req_dir: Path,
        duration: float | None,
        guard: bool,
        decode_s: float | None,
        t0: float | None = None,
    ) -> web.StreamResponse:
        """Transcribe ``chunks`` (one unit for audio that is not chunked), guard, stitch, answer.

        ``decode_s`` is None when the request was not chunked: then a single clean unit's
        engine response goes back as the engine wrote it.
        """
        t0 = time.monotonic() if t0 is None else t0
        rid = req_dir.name[:8]
        headers = _fwd_headers(request.headers)
        headers["Content-Type"] = "application/json"
        prepared = [self._unit_body(body, kind, ch, guard) for ch in chunks]
        caps = [cap for _, cap in prepared]
        results = await asyncio.gather(*(self._post(b, headers) for b, _ in prepared), return_exceptions=True)
        for r in results:
            if isinstance(r, BaseException):
                if isinstance(r, asyncio.CancelledError):
                    raise r
                return _error(502, f"engine request for a chunk failed: {r!r}")
        for status, payload in results:
            if status != 200:
                # The first failing unit's error, as vLLM wrote it.
                return web.Response(status=status, body=payload, content_type="application/json")
        responses = [json.loads(p) for _, p in results]
        report = None
        if guard:
            chunks, responses, report = await self._guard(rid, body, kind, chunks, responses, caps, headers, req_dir)

        extra: dict[str, str] = {}
        if report is not None and report.flagged:
            extra["X-Transcript-Guard"] = report.header()
        if len(chunks) == 1:
            merged, seams = responses[0], []
        else:
            merged, seams = long_audio.merge_chat_responses(responses, chunks, self.cfg.overlap_seconds)
        if report is not None and report.still_anomalous:
            # Visible to the client: this transcript may hold invented or looped text.
            merged["choices"][0]["finish_reason"] = "length"
        if decode_s is not None or len(chunks) > 1:
            extra["X-Audio-Chunks"] = str(len(chunks))
            self._log_chunked(rid, duration, chunks, responses, merged, seams, decode_s, t0)
        if not body.get("stream"):
            return web.json_response(merged, headers=extra)
        return await self._as_sse(request, merged, body, extra)

    def _log_chunked(self, rid, duration, chunks, responses, merged, seams, decode_s, t0) -> None:
        texts = [_content(r) for r in responses]
        finishes = [r["choices"][0].get("finish_reason") for r in responses]
        wps = [runaway.words_per_s(t, c.end_s - c.start_s) for t, c in zip(texts, chunks)]
        log.debug(
            "chunked audio %s: duration=%s chunks=%d spans=%s decode_s=%s total_s=%.2f finish=%s words_per_s=%s usage=%s",
            rid, f"{duration:.1f}" if duration is not None else "unknown", len(chunks),
            [(round(c.start_s, 2), round(c.end_s, 2)) for c in chunks],
            f"{decode_s:.2f}" if decode_s is not None else "-", time.monotonic() - t0,
            finishes, [round(x, 2) for x in wps], merged.get("usage"),
        )
        for i, seam in enumerate(seams):
            log.debug(
                "chunked audio %s: seam %d/%d at %.2f s: matched=%d dropped_prev=%d dropped_next=%d",
                rid, i + 1, len(seams), chunks[i].end_s, seam.matched, seam.dropped_prev, seam.dropped_next,
            )
            if seam.matched == 0 and texts[i].strip() and texts[i + 1].strip():
                # Both chunks transcribed the overlap, differently: one of them is suspect.
                log.debug(
                    "chunked audio %s: seam %d unmatched: chunk %d chars=%d chunk %d chars=%d",
                    rid, i + 1, i, len(texts[i]), i + 1, len(texts[i + 1]),
                )
        typical = sorted(wps)[len(wps) // 2]
        for i, (text, fin, rate) in enumerate(zip(texts, finishes, wps)):
            if fin == "length" or rate < 0.5 * typical:
                log.debug(
                    "chunked audio %s: chunk %d [%.2f, %.2f) finish=%s words_per_s=%.2f chars=%d",
                    rid, i, chunks[i].start_s, chunks[i].end_s, fin, rate, len(text),
                )

    # --- runaway guard ------------------------------------------------------------------

    async def _guard(
        self,
        rid: str,
        body: dict,
        kind: str,
        chunks: list[Chunk],
        responses: list[dict],
        caps: list[int | None],
        headers: dict[str, str],
        req_dir: Path,
    ) -> tuple[list[Chunk], list[dict], GuardReport]:
        """Find units whose transcript ran away, retry each once as two halves, clean up what
        is still anomalous. Returns the units and responses to stitch, and what happened."""
        g = self.guard
        loop = asyncio.get_running_loop()
        texts = [_content(r) for r in responses]
        med = runaway.median([runaway.words_per_s(t, c.end_s - c.start_s) for t, c in zip(texts, chunks)])
        checks = [
            runaway.assess(t, r["choices"][0].get("finish_reason"), c.end_s - c.start_s, cap or 0, g, med)
            for t, r, c, cap in zip(texts, responses, chunks, caps)
        ]
        report = GuardReport()
        bad = [i for i, a in enumerate(checks) if a.anomalous]
        if not bad:
            return chunks, responses, report
        report.flagged = len(bad)
        min_split = g.split_min_seconds if len(chunks) == 1 else g.chunk_split_min_seconds
        for i in bad:
            n_words = checks[i].words
            log.debug(
                "runaway guard %s: unit %d/%d [%.2f, %.2f) anomalous: %s words=%d words_per_s=%.2f median=%s "
                "finish=%s max_tokens=%s chars=%d",
                rid, i, len(chunks), chunks[i].start_s, chunks[i].end_s, "; ".join(checks[i].reasons),
                n_words, checks[i].words_per_s, f"{med:.2f}" if med else "-",
                responses[i]["choices"][0].get("finish_reason"), caps[i], len(texts[i]),
            )
        retry = [i for i in bad if chunks[i].end_s - chunks[i].start_s >= min_split]

        async def halves(i: int) -> tuple[list[Chunk], list[dict], list[runaway.Assessment]] | None:
            try:
                parts: list[Chunk] = await loop.run_in_executor(
                    self.executor, long_audio.split_in_two, chunks[i].path, str(req_dir / f"retry{i}"), self.cfg, chunks[i].start
                )
            except AudioInputError as e:
                log.debug("runaway guard %s: unit %d retry not possible: %s", rid, i, e)
                return None
            prepared = [self._unit_body(body, kind, p, True) for p in parts]
            got = await asyncio.gather(*(self._post(b, headers) for b, _ in prepared), return_exceptions=True)
            out = []
            for r in got:
                if isinstance(r, asyncio.CancelledError):
                    raise r
                if isinstance(r, BaseException):
                    log.debug("runaway guard %s: unit %d retry failed: %s", rid, i, type(r).__name__)
                    return None
                if r[0] != 200:
                    # Status and size only: an engine error body can echo request content (ZDR).
                    log.debug("runaway guard %s: unit %d retry failed: status=%d body_bytes=%d", rid, i, r[0], len(r[1]))
                    return None
                out.append(json.loads(r[1]))
            assessed = [
                runaway.assess(_content(o), o["choices"][0].get("finish_reason"), p.end_s - p.start_s, cap or 0, g, med)
                for o, p, (_, cap) in zip(out, parts, prepared)
            ]
            return parts, out, assessed

        retried = dict(zip(retry, await asyncio.gather(*(halves(i) for i in retry))))
        new_chunks: list[Chunk] = []
        new_resps: list[dict] = []
        still: list[int] = []  # indices into new_* of units still anomalous
        for i, (ch, resp) in enumerate(zip(chunks, responses)):
            got = retried.get(i)
            if got is not None:
                report.retried += 1
                parts, outs, assessed = got
                clean = [not a.anomalous for a in assessed]
                log.debug(
                    "runaway guard %s: unit %d retried as %s: %s",
                    rid, i, [(round(p.start_s, 2), round(p.end_s, 2)) for p in parts],
                    ["ok" if c else "; ".join(a.reasons) for c, a in zip(clean, assessed)],
                )
                if any(clean):
                    # At least half of the unit is now clean: use the halves.
                    report.fixed += all(clean)
                    for p, o, c in zip(parts, outs, clean):
                        if not c:
                            still.append(len(new_chunks))
                        new_chunks.append(p)
                        new_resps.append(o)
                    continue
            if i in bad:
                still.append(len(new_chunks))
            new_chunks.append(ch)
            new_resps.append(resp)
        for k in still:
            text = _content(new_resps[k])
            cleaned = await loop.run_in_executor(self.executor, runaway.cleanup_if_looping, text)
            if cleaned != text:
                report.cleaned += 1
                new_resps[k]["choices"][0]["message"]["content"] = cleaned
            log.debug(
                "runaway guard %s: unit [%.2f, %.2f) still anomalous after retry; loop cleanup %d -> %d words",
                rid, new_chunks[k].start_s, new_chunks[k].end_s, len(text.split()), len(cleaned.split()),
            )
        report.still_anomalous = len(still)
        return new_chunks, new_resps, report

    async def _post(self, body: dict, headers: dict[str, str]) -> tuple[int, bytes]:
        assert self.session is not None
        data = json.dumps(body).encode()
        attempts = 2 if self.retry_transient else 1
        for attempt in range(1, attempts + 1):
            try:
                async with self.session.post(self.upstream + CHAT_PATH, data=data, headers=headers) as r:
                    return r.status, await r.read()
            except (aiohttp.ServerDisconnectedError, aiohttp.ClientOSError) as e:
                if attempt == attempts:
                    raise
                log.debug("transient_retry engine attempt=%d error=%r", attempt, e)
        raise AssertionError("unreachable")

    async def _as_sse(self, request: web.Request, merged: dict, body: dict, extra: dict) -> web.StreamResponse:
        """A finished chunked completion as the ``chat.completion.chunk`` events a stream ends with."""
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache", **extra})
        await resp.prepare(request)
        choice = merged["choices"][0]
        base = {"id": merged.get("id"), "object": "chat.completion.chunk", "created": merged.get("created"), "model": merged.get("model")}

        async def send(obj: dict) -> None:
            await resp.write(b"data: " + json.dumps(obj).encode() + b"\n\n")

        await send({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": choice["message"].get("content") or ""}, "finish_reason": None}]})
        final = {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": choice.get("finish_reason")}]}
        if "ctc_timestamps" in merged:
            final["ctc_timestamps"] = merged["ctc_timestamps"]
        await send(final)
        if (body.get("stream_options") or {}).get("include_usage") and merged.get("usage") is not None:
            await send({**base, "choices": [], "usage": merged["usage"]})
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp


def _flag(env, name: str) -> bool:
    raw = env.get(name, "") or "0"
    if raw not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1, got {raw!r}")
    return raw == "1"


def build(env=None) -> tuple[ChunkConfig, dict[str, Any]]:
    env = os.environ if env is None else env
    cfg = ChunkConfig.from_env(env)
    workers_raw = env.get("NEMOTRON_CHUNK_DECODE_WORKERS", "") or "2"
    if not workers_raw.isdigit() or int(workers_raw) < 1:
        raise ValueError(f"NEMOTRON_CHUNK_DECODE_WORKERS must be a positive integer, got {workers_raw!r}")
    opts = {
        "workers": int(workers_raw),
        "work_dir": env.get("NEMOTRON_CHUNK_DIR", "") or "/tmp/nemotron-chunks",
        "fetch_timeout_s": float(env.get("NEMOTRON_AUDIO_FETCH_TIMEOUT", "") or 300),
        "guard": GuardConfig.from_env(env),
        "retry_transient": _flag(env, "NEMOTRON_PROXY_RETRY"),
    }
    if cfg.enabled or opts["guard"].enabled:
        import av  # noqa: F401  (fail at start, not on the first long request)
    return cfg, opts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="validate configuration and exit")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--upstream", default="http://127.0.0.1:8001")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[chunk_proxy] %(levelname)s %(message)s")
    try:
        # Only this proxy's logger follows NEMOTRON_LOG_LEVEL; aiohttp and the rest stay at INFO.
        log.setLevel(log_level())
        cfg, opts = build()
    except (ValueError, ImportError) as e:
        print(f"chunk_proxy: invalid configuration: {e}", file=sys.stderr)
        return 1
    log.info("config %s %s", cfg, opts)
    if args.check:
        return 0
    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=opts["workers"], mp_context=multiprocessing.get_context("spawn")
    )
    proxy = ChunkProxy(
        cfg, args.upstream, opts["work_dir"], executor, opts["fetch_timeout_s"], opts["guard"],
        opts["retry_transient"],
    )
    # handler_cancellation: a client that disconnects cancels its chunk requests, so vLLM aborts them.
    web.run_app(proxy.app(), host=args.host, port=args.port, access_log=None, handler_cancellation=True, print=None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
