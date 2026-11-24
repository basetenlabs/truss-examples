"""VibeVoice-ASR long-form proxy.

Sits on the public port in front of vLLM (which moves to 127.0.0.1):

- every path passes through unchanged (``/v1/models``, ``/health``, ``/metrics``, ...);
- ``POST /v1/chat/completions`` passes through; http(s) or non-WAV/FLAC audio is first
  decoded with ffmpeg to 24 kHz mono WAV (the plugin's stdin-pipe decode cannot read
  m4a). With the levers on (``VV_CHUNK_S``, ``VV_MAX_TOKENS_PER_S``,
  ``VV_CANONICAL_PROMPT``; the preset ships all three on, see config.py) the proxy rewrites the
  request: the upstream prompt recipe (hotwords kept), a loop guard that only lowers ``max_tokens``,
  and chunking of long non-streaming audio;
- ``POST /v1/audio/transcriptions`` (multipart ``file``) returns OpenAI ``verbose_json``
  with a ``speaker`` per segment, for diarization scoring (tcpWER/cpWER);
- one ``timing`` log line per audio request;
- pooled vLLM connections expire before uvicorn's keep-alive and a closed one is retried once on
  a fresh connection; remote audio fetch failures are 502, not "engine not reachable" 503.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import aiohttp
from aiohttp import web

from . import audio, chunking, linking, transcript
from .config import Settings

# Upstream recipe (microsoft/VibeVoice@16fb2cb): vibevoice/processor/vibevoice_asr_processor.py:27,396 and
# vllm_plugin/tests/test_api.py:150-175 (system message, then a user turn with the audio part first and
# the duration text after it).
SYSTEM_PROMPT = (
    "You are a helpful assistant that transcribes audio input into text output in JSON format."
)
# Stop tokens are not set here: data/start.sh writes generation_config.json with eos_token_id
# [151643 <|endoftext|>, 151645 <|im_end|>] next to the tokenizer files, and vLLM adds both to every
# request's stop tokens, on every route and with VV_PROXY=0 (vllm 0.14.1 SamplingParams
# .update_from_generation_config, called from v1/engine/input_processor.py).
# Hotwords / context, as upstream's vllm_plugin/asr_streaming_server.py (_CONTEXT_RE) reads them from the
# client's text parts; the processor puts them in the prompt as "with extra info: ..."
# (vibevoice/processor/vibevoice_asr_processor.py:393-394).
_CONTEXT_RE = re.compile(
    r"context information[^\n:]*:[ \t]*\n(.+)|with extra info:[ \t]*([^\n]+)",
    re.IGNORECASE | re.DOTALL)
DEFAULT_TEXT = "Transcribe this audio."  # what b10-bench openai_chat_audio sends today
AUDIO_TOKENS_PER_S = 7.5  # 24 kHz / 3200 samples per token
PROMPT_TEXT_TOKENS = 128  # headroom for chat template + text prompt
MIN_MAX_TOKENS = 256

_HOP = {"host", "content-length", "transfer-encoding", "connection", "keep-alive",
        "content-encoding", "upgrade", "te", "trailer", "proxy-connection"}  # fmt: skip
MAX_BODY = 1024 * 1024 * 1024  # 1 GiB; Baseten ingress caps bodies at 100 MB anyway

# Pooled connections to vLLM expire after this many idle seconds, below uvicorn's keep-alive
# (vLLM's VLLM_HTTP_TIMEOUT_KEEP_ALIVE, default 5 s). The age check is by clock, so it holds even
# when this event loop was busy in synchronous prep (WAV encode, json.dumps of a 100+ MB body)
# and never saw vLLM's FIN; aiohttp's default (15 s) reused such closed connections and the next
# write failed with EPIPE ("[Errno 32] Broken pipe") or the read with "Server disconnected".
UPSTREAM_KEEPALIVE_S = 2.0
# Errors on an established (pooled) connection that closed under us. ClientConnectorError (connect
# refused: engine down) is not retried.
STALE_CONN_ERRORS = (aiohttp.ServerDisconnectedError, aiohttp.ClientOSError)
# Stands in for a chunk's audio URL in its JSON body; the base64 WAV is spliced in as bytes, which
# gives the same bytes as json.dumps with the real URL (base64 needs no JSON escaping).
AUDIO_PLACEHOLDER = "@@vv_longform_chunk_audio@@"


def canonical_text(duration: float, context: str | None = None) -> str:
    keys = "Start time, End time, Speaker ID, Content"
    if context and context.strip():
        return (f"This is a {duration:.2f} seconds audio, with extra info: {context.strip()}\n\n"
                f"Please transcribe it with these keys: {keys}")  # fmt: skip
    return f"This is a {duration:.2f} seconds audio, please transcribe it with these keys: {keys}"


def context_from_messages(messages) -> str | None:
    """The hotword block of the client's text parts, as upstream ``_context_from_messages`` finds it."""
    for msg in messages or []:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if not (isinstance(part, dict) and part.get("type") == "text"):
                continue
            m = _CONTEXT_RE.search(part.get("text") or "")
            if m:
                found = (m.group(1) or m.group(2) or "").strip()
                if found:
                    return found
    return None


def client_max_tokens(body: dict) -> int | None:
    """The request's own ``max_tokens`` when it is a positive integer (500.0 counts), else None."""
    v = body.get("max_tokens")
    if isinstance(v, bool):
        return None
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return v if isinstance(v, int) and v > 0 else None


@dataclass
class Timing:
    route: str
    t0: float = field(default_factory=time.perf_counter)
    status: int = 0
    audio_s: float | None = None
    n_chunks: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    n_length: int = 0
    max_tokens: int | None = None
    prep_ms: float = 0.0
    queue_ms: float = 0.0
    gen_ms: float = 0.0
    link_ms: float = 0.0
    link: str | None = None  # set only for multi-chunk requests
    n_speakers: int | None = None
    embed_ms: float = 0.0
    fetch_ms: float = 0.0
    decode: audio.DecodeInfo | None = None  # set when the chunked-request decode ran
    error: str | None = None
    upstream_retries: int = 0

    def line(self) -> str:
        total = (time.perf_counter() - self.t0) * 1000

        def f(v):
            if v is None:
                return "na"
            return f"{v:.1f}" if isinstance(v, float) else str(v)

        parts = [
            f"route={self.route}", f"status={self.status}", f"audio_s={f(self.audio_s)}",
            f"n_chunks={self.n_chunks}", f"prompt_tokens={f(self.prompt_tokens)}",
            f"completion_tokens={f(self.completion_tokens)}",
            f"finish_reason={self.finish_reason or 'na'}", f"n_length={self.n_length}",
            f"max_tokens={f(self.max_tokens)}", f"prep_ms={self.prep_ms:.1f}",
            f"queue_ms={self.queue_ms:.1f}", f"gen_ms={self.gen_ms:.1f}",
            f"link_ms={self.link_ms:.1f}", f"total_ms={total:.1f}",
        ]  # fmt: skip
        if self.link is not None:
            parts += [f"link={self.link}", f"n_speakers={f(self.n_speakers)}", f"embed_ms={self.embed_ms:.1f}"]
        if self.fetch_ms:
            parts.append(f"fetch_ms={self.fetch_ms:.1f}")
        if self.decode is not None:
            d = self.decode
            parts += [f"decode_ms={d.decode_ms:.1f}", f"prep_mode={d.mode}", f"n_seg={d.n_segments}"]
            if d.seam_mismatch:
                parts.append(f"seam_mismatch={d.seam_mismatch}")
        if self.upstream_retries:
            parts.append(f"upstream_retries={self.upstream_retries}")
        if self.error:
            parts.append("error=" + json.dumps(self.error[:200]))
        return "timing " + " ".join(parts)


class UpstreamError(Exception):
    def __init__(self, status: int, body: bytes, content_type: str = "application/json"):
        super().__init__(f"upstream HTTP {status}")
        self.status, self.body, self.content_type = status, body, content_type


def _is_stale(exc: BaseException) -> bool:
    return isinstance(exc, STALE_CONN_ERRORS) and not isinstance(exc, aiohttp.ClientConnectorError)


def _json_error(status: int, message: str) -> web.Response:
    body = {"error": {"message": message, "type": "BadRequestError" if status < 500 else "ServerError",
                      "code": status}}  # fmt: skip
    return web.json_response(body, status=status)


def _audio_parts(body: dict) -> list[dict]:
    """The ``audio_url`` content parts of a chat request (dicts with a ``url``)."""
    out = []
    for msg in body.get("messages") or []:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "audio_url":
                au = part.get("audio_url")
                if isinstance(au, dict) and isinstance(au.get("url"), str):
                    out.append(au)
    return out


class Proxy:
    def __init__(self, settings: Settings, embedder=None):
        self.s = settings
        self.session: aiohttp.ClientSession | None = None  # to vLLM
        self.fetch_session: aiohttp.ClientSession | None = None  # remote audio URLs
        self.chunk_sem = asyncio.Semaphore(max(1, settings.max_inflight_chunks))
        # Speaker linking (VV_LINK=embed with VV_CHUNK_S>0): the embedding model loads on the
        # CPU at startup, moves to the GPU once vLLM answers, and /v1/models + /health
        # return 503 until it is ready. ``embedder`` injects a ready one (tests).
        self.embedder = embedder
        self.link_ready = asyncio.Event()
        self._link_pool: ThreadPoolExecutor | None = None
        self._link_task: asyncio.Task | None = None
        # Blocking audio prep (temp-file writes, ffmpeg decodes, cut search) and payload encodes
        # (WAV + base64) run here, never on the event loop.
        self._prep_pool = ThreadPoolExecutor(max_workers=settings.prep_threads + 1,
                                             thread_name_prefix="vv_prep")  # fmt: skip
        self._encode_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vv_encode")

    # --- lifecycle -----------------------------------------------------------------------------

    async def start(self, app: web.Application) -> None:
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=10),
            connector=aiohttp.TCPConnector(limit=0, keepalive_timeout=UPSTREAM_KEEPALIVE_S),
        )
        # Remote audio fetches are rare, large downloads from hosts whose keep-alive we do not
        # know: one connection per fetch, so a pooled connection the host closed is never reused.
        self.fetch_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=120),
            connector=aiohttp.TCPConnector(limit=0, force_close=True),
        )
        if self.s.link_embed:
            self._link_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vv_link")
            if self.embedder is not None:
                self.link_ready.set()
            else:
                self._link_task = asyncio.create_task(self._load_linker())

    async def stop(self, app: web.Application) -> None:
        if self._link_task is not None:
            self._link_task.cancel()
        if self._link_pool is not None:
            self._link_pool.shutdown(wait=False, cancel_futures=True)
        self._prep_pool.shutdown(wait=False, cancel_futures=True)
        self._encode_pool.shutdown(wait=False, cancel_futures=True)
        if self.session is not None:
            await self.session.close()
        if self.fetch_session is not None:
            await self.fetch_session.close()

    async def _load_linker(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            t0 = time.perf_counter()

            def load():
                from .embed import SpeakerEmbedder  # torch import happens here, off the default path

                return SpeakerEmbedder(self.s.link_model_path, self.s.link_device,
                                       self.s.link_gpu_mem_fraction, self.s.link_cpu_threads)  # fmt: skip

            emb = await loop.run_in_executor(self._link_pool, load)
            t1 = time.perf_counter()
            # Touch the GPU only after vLLM has sized its memory pool.
            assert self.session is not None
            while True:
                try:
                    async with self.session.get(self.s.upstream + "/v1/models") as r:
                        if r.status == 200:
                            break
                except aiohttp.ClientError:
                    pass
                await asyncio.sleep(1.0)
            t2 = time.perf_counter()
            dev = await loop.run_in_executor(self._link_pool, emb.to_device)
            t3 = time.perf_counter()
            self.embedder = emb
            self.link_ready.set()
            print(f"[vv_longform] speaker linker ready on {dev}: cpu_load_ms={(t1 - t0) * 1000:.0f} "
                  f"to_device_ms={(t3 - t2) * 1000:.0f} (after vLLM ready)", file=sys.stderr, flush=True)  # fmt: skip
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 - fail the replica loudly, never serve unlinked
            print(f"[vv_longform] speaker linker failed to load: {exc!r}; exiting", file=sys.stderr, flush=True)
            os._exit(1)

    def app(self) -> web.Application:
        app = web.Application(client_max_size=MAX_BODY)
        app.on_startup.append(self.start)
        app.on_cleanup.append(self.stop)
        app.router.add_post("/v1/chat/completions", self.chat)
        app.router.add_post("/v1/audio/transcriptions", self.transcriptions)
        app.router.add_route("*", "/{tail:.*}", self.passthrough)
        return app

    def _log(self, t: Timing) -> None:
        if self.s.log_timing:
            print(t.line(), flush=True)

    # --- upstream requests ---------------------------------------------------------------------

    @contextlib.asynccontextmanager
    async def _upstream(self, method: str, url: str, t: Timing | None = None, **kw):
        """``session.request`` to vLLM that retries once, on a fresh unpooled connection, when the
        pooled connection turns out to be closed (EPIPE/ECONNRESET writing the body, or the
        server hanging up before the response headers). Nothing has reached the client yet, and
        ``data`` must be bytes so it can be resent."""
        assert self.session is not None
        fresh = None
        try:
            try:
                resp = await self.session.request(method, url, **kw)
            except STALE_CONN_ERRORS as exc:
                if not _is_stale(exc):
                    raise
                if t is not None:
                    t.upstream_retries += 1
                print(f"[vv_longform] upstream {method} {url[len(self.s.upstream):][:60]}: stale "
                      f"connection ({exc!r}); retrying once on a fresh connection",
                      file=sys.stderr, flush=True)  # fmt: skip
                fresh = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=None, sock_connect=10),
                    connector=aiohttp.TCPConnector(force_close=True),
                )
                resp = await fresh.request(method, url, **kw)
            try:
                yield resp
            finally:
                resp.release()
        finally:
            if fresh is not None:
                await fresh.close()

    # --- plain passthrough ---------------------------------------------------------------------

    async def passthrough(self, request: web.Request, body: bytes | None = None) -> web.StreamResponse:
        assert self.session is not None
        if (self.s.link_embed and not self.link_ready.is_set()
                and request.method == "GET" and request.path in ("/v1/models", "/health")):
            return _json_error(503, "speaker linker loading")
        if body is None:
            body = await request.read()
        headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP}
        url = self.s.upstream + str(request.rel_url)
        try:
            async with self._upstream(request.method, url, headers=headers, data=body) as up:
                resp = web.StreamResponse(status=up.status)
                for k, v in up.headers.items():
                    if k.lower() not in _HOP:
                        resp.headers[k] = v
                await resp.prepare(request)
                async for chunk in up.content.iter_any():
                    await resp.write(chunk)
                await resp.write_eof()
                return resp
        except aiohttp.ClientConnectionError as exc:
            return _json_error(503, f"engine not reachable: {exc}")

    async def _post_chat(self, body: dict, t: Timing | None = None, data: bytes | None = None) -> dict:
        """POST ``data`` (default: ``json.dumps(body)``) to vLLM's chat route."""
        async with self._upstream("POST", self.s.upstream + "/v1/chat/completions", t,
                                  data=data if data is not None else json.dumps(body).encode(),
                                  headers={"Content-Type": "application/json"}) as up:  # fmt: skip
            raw = await up.read()
            if up.status != 200:
                raise UpstreamError(up.status, raw, up.headers.get("Content-Type", "application/json"))
            return json.loads(raw)

    # --- audio preparation ---------------------------------------------------------------------

    async def _fetch(self, url: str) -> bytes:
        """GET a remote audio URL; one retry on a dropped connection (GET is idempotent).
        Every aiohttp failure (connection, redirects, bad URL, payload) and a timeout raise
        ``audio.FetchError`` (HTTP 502), not "engine not reachable". The error names only the
        exception type and drops the cause: aiohttp's errors carry the signed URL (ZDR)."""
        assert self.fetch_session is not None
        for attempt in (0, 1):
            try:
                async with self.fetch_session.get(url) as r:
                    if r.status != 200:
                        raise audio.AudioError(f"audio fetch HTTP {r.status}")  # no URL: signed query (ZDR)
                    return await r.read()
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt == 0 and _is_stale(exc):
                    continue
                raise audio.FetchError(f"audio fetch failed: {type(exc).__name__}") from None
        raise AssertionError("unreachable")

    async def _decode(self, data: bytes, need_pcm: bool, t: Timing | None):
        """``audio.decode`` for single-pass requests; the pooled (and, for long audio, parallel)
        ``audio.decode_long`` when the request may be chunked. Both give the same samples unless
        VV_PREP_SEAM_FALLBACK=0 keeps a mismatching parallel decode."""
        if not need_pcm:
            return await audio.decode(data)
        pcm, info = await audio.decode_long(
            data, self._prep_pool, parallel=self.s.prep_parallel,
            min_parallel_s=self.s.effective_chunk_trigger_s, workers=self.s.prep_threads,
            min_seg_s=self.s.prep_min_seg_s, fallback=self.s.prep_seam_fallback,
        )  # fmt: skip
        if t is not None:
            t.decode = info
        return pcm

    async def _wav_uri(self, pcm) -> str:
        return await asyncio.get_running_loop().run_in_executor(self._encode_pool, audio.wav_data_uri, pcm)

    async def _load(self, url: str, need_pcm: bool, stream: bool = False, t: Timing | None = None):
        """(new_url or None, pcm or None, audio_s or None) for one audio_url. A request that will
        be chunked gets no whole-file WAV URL (each chunk gets its own), so new_url is None."""
        if url.startswith("data:"):
            mime, data = audio.split_data_uri(url)
            dur = audio.wav_duration_s(data)
            native = dur is not None or audio.is_flac(data)
            if dur is not None and not (need_pcm and self._chunking(dur)):
                return None, None, dur  # WAV: untouched, duration from the header
            if native and not need_pcm and not self.s.decode_remote_audio:
                return None, None, None
            pcm = await self._decode(data, need_pcm, t)
            dur = pcm.shape[0] / audio.SAMPLE_RATE
            if native or (need_pcm and self._chunking(dur) and not stream):
                return None, pcm, dur
            return await self._wav_uri(pcm), pcm, dur
        if url.startswith(("http://", "https://")) and (self.s.decode_remote_audio or need_pcm):
            f0 = time.perf_counter()
            data = await self._fetch(url)
            if t is not None:
                t.fetch_ms += (time.perf_counter() - f0) * 1000
            pcm = await self._decode(data, need_pcm, t)
            dur = pcm.shape[0] / audio.SAMPLE_RATE
            if need_pcm and self._chunking(dur) and not stream:
                return None, pcm, dur
            return await self._wav_uri(pcm), pcm, dur
        return None, None, None

    def _max_tokens_rule(self, dur: float, fallback: int | None) -> int | None:
        """Lever 2 when set, else ``fallback`` (None = leave the request's own value)."""
        if self.s.max_tokens_per_s is not None:
            mt = int(math.ceil(self.s.max_tokens_per_s * dur)) + self.s.max_tokens_base
        elif fallback is not None:
            mt = fallback
        else:
            return None
        return max(MIN_MAX_TOKENS, min(self.s.max_tokens_cap, mt))

    def _window_cap(self, dur: float) -> int:
        """Largest max_tokens that fits the context window next to ``dur`` s of audio."""
        return self.s.max_model_len - int(math.ceil(dur * AUDIO_TOKENS_PER_S)) - PROMPT_TEXT_TOKENS

    def _apply_prompt(self, body: dict, dur: float) -> None:
        """VV_CANONICAL_PROMPT=1: rewrite to the upstream recipe. Any client system message is
        replaced by SYSTEM_PROMPT; each user turn keeps its audio parts, in order, followed by one
        ``canonical_text(dur, context)`` part, where ``context`` is the client's hotword block
        ("with extra info: ..."), if any."""
        if not self.s.canonical_prompt:
            return
        context = context_from_messages(body.get("messages"))
        msgs = [m for m in (body.get("messages") or [])
                if not (isinstance(m, dict) and m.get("role") == "system")]  # fmt: skip
        msgs.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        for m in msgs:
            if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), list):
                kept = [p for p in m["content"] if not (isinstance(p, dict) and p.get("type") == "text")]
                m["content"] = kept + [{"type": "text", "text": canonical_text(dur, context)}]
        body["messages"] = msgs

    def _chunking(self, dur: float | None) -> bool:
        return bool(self.s.chunk_s > 0 and dur is not None and dur > self.s.effective_chunk_trigger_s)

    # --- generation over chunks ----------------------------------------------------------------

    async def _run_chunks(
        self, template: dict, pcm, t: Timing, chunk_s: float
    ) -> tuple[list[transcript.Segment], dict]:
        """Fan ``pcm`` out as ``chunk_s`` chunk requests built from ``template`` (0 = one
        request for the whole file) and stitch the replies."""
        rate = audio.SAMPLE_RATE
        loop = asyncio.get_running_loop()
        cuts = await loop.run_in_executor(self._prep_pool, chunking.bounds, pcm, rate, chunk_s,
                                          self.s.cut_search_s)  # fmt: skip
        t.n_chunks = len(cuts)
        waits: list[float] = []
        link_on = self.s.link_embed and len(cuts) > 1
        if len(cuts) > 1:
            t.link = "embed" if link_on else "none"
        embed_ms: list[float] = []

        embed_tasks: list[asyncio.Future] = []

        async def one(i: int, a: int, b: int):
            dur = (b - a) / rate
            body = copy.deepcopy(template)
            aps = _audio_parts(body)
            for au in aps:
                au["url"] = AUDIO_PLACEHOLDER
            self._apply_prompt(body, dur)
            mt = self._max_tokens_rule(dur, int(20 * dur + 200))
            body["max_tokens"] = max(1, min(mt, self._window_cap(dur)))
            head = json.dumps(body)

            def build() -> bytes:
                pieces = head.encode().split(AUDIO_PLACEHOLDER.encode())
                uri = audio.WAV_URI_PREFIX + audio.wav_b64(pcm[a:b])
                if len(pieces) == len(aps) + 1:
                    return uri.join(pieces)
                # The placeholder also occurs elsewhere in the request: build it the slow way.
                for au in aps:
                    au["url"] = uri.decode("ascii")
                return json.dumps(body).encode()

            data = await loop.run_in_executor(self._encode_pool, build)
            q0 = time.perf_counter()
            async with self.chunk_sem:
                waits.append((time.perf_counter() - q0) * 1000)
                resp = await self._post_chat(body, t, data)
            choice = (resp.get("choices") or [{}])[0]
            segs = transcript.parse((choice.get("message") or {}).get("content"))
            # Start embedding this chunk's speakers while the other chunks still decode.
            task = None
            if link_on:
                task = asyncio.ensure_future(self._embed_chunk(i, a, (b - a) / rate, pcm, segs, embed_ms))
                embed_tasks.append(task)
            return resp, body["max_tokens"], segs, task

        g0 = time.perf_counter()
        tasks = [asyncio.ensure_future(one(i, a, b)) for i, (a, b) in enumerate(cuts)]
        try:
            results = await asyncio.gather(*tasks)
        except BaseException:
            # One chunk failed (or the client went away): stop the others generating on the GPU and
            # drop their speaker embeddings, then re-raise the first error.
            pending = [x for x in tasks + embed_tasks if not x.done()]
            for x in pending:
                x.cancel()
            await asyncio.gather(*tasks, *embed_tasks, return_exceptions=True)
            raise
        t.gen_ms = (time.perf_counter() - g0) * 1000
        t.queue_ms = max(waits) if waits else 0.0
        l0 = time.perf_counter()
        parts, usage, finishes = [], {"prompt_tokens": 0, "completion_tokens": 0}, []
        for (a, b), (resp, _, csegs, _) in zip(cuts, results):
            choice = (resp.get("choices") or [{}])[0]
            parts.append((a / rate, (b - a) / rate, csegs))
            u = resp.get("usage") or {}
            usage["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
            usage["completion_tokens"] += int(u.get("completion_tokens") or 0)
            finishes.append(choice.get("finish_reason"))
        labels = None
        if link_on:
            items = [it for r in await asyncio.gather(*(r[3] for r in results)) for it in r]
            # Clustering is cubic in the number of chunk speakers: keep it off the event loop.
            labels = await loop.run_in_executor(self._link_pool, linking.labels, items,
                                                self.s.link_threshold, self.s.link_min_s)  # fmt: skip
            t.embed_ms = sum(embed_ms)
        segs = transcript.stitch(parts, labels)
        if len(cuts) > 1:
            t.n_speakers = len({s.speaker for s in segs if s.speaker != "unknown"})
        t.link_ms = (time.perf_counter() - l0) * 1000
        t.n_length = sum(1 for f in finishes if f == "length")
        t.finish_reason = "length" if t.n_length else (finishes[0] if len(set(finishes)) == 1 else "stop")
        t.prompt_tokens, t.completion_tokens = usage["prompt_tokens"], usage["completion_tokens"]
        t.max_tokens = sum(r[1] for r in results)
        merged = dict(results[0][0])
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        merged["usage"] = usage
        merged["choices"] = [{
            "index": 0, "finish_reason": t.finish_reason, "logprobs": None,
            "message": {"role": "assistant", "content": transcript.to_model_json(segs)},
        }]  # fmt: skip
        return segs, merged

    async def _embed_chunk(self, i: int, a: int, length: float, pcm, segs: list[transcript.Segment],
                           embed_ms: list[float]) -> list[linking.Item]:
        """Embed each local speaker of chunk ``i`` (starting at sample ``a``)."""
        rate = audio.SAMPLE_RATE
        ranges = linking.speaker_ranges(segs, length, self.s.link_max_s)
        if not ranges:
            return []
        await self.link_ready.wait()
        items = linking.make_items(i, a / rate, length, segs, ranges)
        clips = [linking.gather(pcm, rate, a, ranges[it.speaker]) for it in items]

        def run():
            t0 = time.perf_counter()
            out = []
            for clip in clips:
                # < 0.2 s cannot give a usable embedding; such a speaker keeps its own label.
                try:
                    out.append(self.embedder.embed_pcm24k(clip) if clip.size >= 0.2 * rate else None)
                except ValueError as exc:
                    print(f"[vv_longform] speaker embedding failed: {exc}", file=sys.stderr, flush=True)
                    out.append(None)
            embed_ms.append((time.perf_counter() - t0) * 1000)
            return out

        loop = asyncio.get_running_loop()
        for it, e in zip(items, await loop.run_in_executor(self._link_pool, run)):
            it.emb = e
        return items

    # --- routes --------------------------------------------------------------------------------

    async def chat(self, request: web.Request) -> web.StreamResponse:
        t = Timing("chat")
        raw = await request.read()
        try:
            body = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return await self.passthrough(request, raw)
        parts = _audio_parts(body) if isinstance(body, dict) else []
        if not parts:
            return await self.passthrough(request, raw)
        try:
            changed = False
            pcm, dur = None, None
            for au in parts:
                need_pcm = len(parts) == 1 and self.s.chunk_s > 0
                new_url, p, d = await self._load(au["url"], need_pcm, bool(body.get("stream")), t)
                if new_url is not None:
                    au["url"], changed = new_url, True
                if len(parts) == 1:
                    pcm, dur = p, d
            t.audio_s = dur
            t.prep_ms = (time.perf_counter() - t.t0) * 1000
            if self._chunking(dur) and pcm is not None and not body.get("stream"):
                _, merged = await self._run_chunks(body, pcm, t, self.s.chunk_s)
                t.status = 200
                return web.json_response(merged)
            t.n_chunks = 1
            if dur is not None and (self.s.canonical_prompt or self.s.max_tokens_per_s is not None):
                self._apply_prompt(body, dur)
                mt = self._max_tokens_rule(dur, None)
                if mt is not None:
                    # The loop guard only lowers the client's max_tokens, never raises it.
                    own = client_max_tokens(body)
                    body["max_tokens"] = max(1, min(mt, self._window_cap(dur), own if own is not None else mt))
                changed = True
            t.max_tokens = body.get("max_tokens") if isinstance(body.get("max_tokens"), int) else None
            if body.get("stream"):
                return await self.passthrough(request, json.dumps(body).encode() if changed else raw)
            g0 = time.perf_counter()
            async with self._upstream(
                "POST", self.s.upstream + "/v1/chat/completions", t,
                data=json.dumps(body).encode() if changed else raw,
                headers={"Content-Type": "application/json"},
            ) as up:
                out = await up.read()
                t.gen_ms = (time.perf_counter() - g0) * 1000
                t.status = up.status
                if up.status == 200:
                    try:
                        r = json.loads(out)
                        u = r.get("usage") or {}
                        t.prompt_tokens, t.completion_tokens = u.get("prompt_tokens"), u.get("completion_tokens")
                        t.finish_reason = (r.get("choices") or [{}])[0].get("finish_reason")
                        t.n_length = int(t.finish_reason == "length")
                    except (json.JSONDecodeError, AttributeError, IndexError):
                        pass
                else:
                    t.error = f"upstream HTTP {up.status}"  # the body may echo request content (ZDR)
                return web.Response(status=up.status, body=out,
                                    content_type=up.content_type or "application/json")  # fmt: skip
        except audio.FetchError as exc:
            t.status, t.error = 502, str(exc)
            return _json_error(502, str(exc))
        except audio.AudioError as exc:
            t.status, t.error = 400, str(exc)
            return _json_error(400, str(exc))
        except UpstreamError as exc:
            t.status, t.error = exc.status, f"upstream HTTP {exc.status}"  # body may echo content (ZDR)
            return web.Response(status=exc.status, body=exc.body, content_type="application/json")
        except aiohttp.ClientConnectionError as exc:
            t.status, t.error = 503, str(exc)
            return _json_error(503, f"engine not reachable: {exc}")
        finally:
            self._log(t)

    async def transcriptions(self, request: web.Request) -> web.StreamResponse:
        t = Timing("transcriptions")
        try:
            fields: dict[str, str] = {}
            data = None
            reader = await request.multipart()
            async for part in reader:
                if part.name == "file":
                    data = await part.read(decode=False)
                elif part.name:
                    fields[part.name] = (await part.text()).strip()
            if data is None:
                t.status = 400
                return _json_error(400, "multipart field 'file' is required")
            pcm = await self._decode(bytes(data), self.s.chunk_s > 0, t)
            dur = pcm.shape[0] / audio.SAMPLE_RATE
            t.audio_s = dur
            t.prep_ms = (time.perf_counter() - t.t0) * 1000
            fmt = fields.get("response_format", "json")
            template: dict = {
                "model": self.s.served_model,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": DEFAULT_TEXT},
                    {"type": "audio_url", "audio_url": {"url": ""}},
                ]}],
                "temperature": float(fields.get("temperature") or 0.0),
            }  # fmt: skip
            chunk_s = self.s.chunk_s if self._chunking(dur) else 0.0
            if chunk_s == 0.0:
                # One pass through the same code path as chunks (one chunk = whole file).
                if self._window_cap(dur) < MIN_MAX_TOKENS:
                    t.status = 400
                    msg = (f"{dur:.0f} s of audio leaves no room for output in VV_MAX_MODEL_LEN="
                           f"{self.s.max_model_len}; raise it or enable VV_CHUNK_S")  # fmt: skip
                    t.error = msg
                    return _json_error(400, msg)
            segs, merged = await self._run_chunks(template, pcm, t, chunk_s)
            t.status = 200
            if fmt == "text":
                return web.Response(text=" ".join(s.text for s in segs if s.text))
            out = transcript.to_verbose_json(segs, dur)
            if fmt != "verbose_json":
                out = {"text": out["text"]}
            out["finish_reason"] = t.finish_reason
            out["usage"] = merged.get("usage")
            return web.json_response(out)
        except audio.FetchError as exc:
            t.status, t.error = 502, str(exc)
            return _json_error(502, str(exc))
        except audio.AudioError as exc:
            t.status, t.error = 400, str(exc)
            return _json_error(400, str(exc))
        except UpstreamError as exc:
            t.status, t.error = exc.status, f"upstream HTTP {exc.status}"  # body may echo content (ZDR)
            return web.Response(status=exc.status, body=exc.body, content_type="application/json")
        except aiohttp.ClientConnectionError as exc:
            t.status, t.error = 503, str(exc)
            return _json_error(503, f"engine not reachable: {exc}")
        finally:
            self._log(t)


def main(settings: Settings) -> None:
    proxy = Proxy(settings)
    print(f"[vv_longform] listening {settings.listen_host}:{settings.listen_port} -> {settings.upstream} "
          f"settings={settings}", file=sys.stderr, flush=True)  # fmt: skip
    web.run_app(proxy.app(), host=settings.listen_host, port=settings.listen_port,
                access_log=None, print=None)  # fmt: skip
