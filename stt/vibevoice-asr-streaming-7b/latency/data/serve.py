#!/usr/bin/env python3
"""Boot upstream's streaming ASR server with one extra WebSocket route.

Why this file exists
--------------------
`docker_server` pass-through routes exactly one container path: whatever
`predict_endpoint` names. Upstream's own socket, `/v1/stream`, therefore
cannot be both the routed socket AND speak a dialect b10-bench understands:

  * upstream `/v1/stream` — JSON config frame, then binary little-endian
    float32 PCM at the checkpoint's 24 kHz, then a text frame `"end"`;
    replies `{"chunk", "text", "segments"}` per window and `{"done", ...}` last.
  * b10-bench `qwen_realtime` — text JSON only: an optional handshake, then
    `{"type": "input_audio_buffer.append", "audio": "<base64 PCM16 @16kHz>"}`
    frames, then `{"type": "input_audio_buffer.commit"}`; expects
    `{"type": "transcription", "is_final": ..., "segments": [{start_time,
    end_time, text}]}` back, the final one carrying `is_end_of_audio_flush`.
    (Server side of that contract, in this repo:
    `stt/qwen3-asr-1.7b-streaming/latency/model/model.py`.)

Nothing bridges those, so this launcher mounts a translator at
`/v1/realtime` — the container path `docker_server.predict_endpoint` names, so
Baseten's WebSocket transport proxies the gateway's fixed `/websocket` route
straight to it. (Same container path `stt/voxtral-mini-4b/latency` uses, and
the one the realtime protocols' own `/sync/v1/realtime` default resolves to on
a plain-HTTP truss; here the transport decides, not that default.)

The translator is a protocol shim only. It does not re-implement the session:
it drives upstream's own `StreamingSession` through `_new_session`, cuts
windows with upstream's `split_windows`, and renders text with upstream's
`chunk_segments`, exactly as `/v1/stream` does. What it adds is (a) the
qwen_realtime dialect and (b) a streaming 16 kHz -> 24 kHz resampler, because
every b10-bench STT protocol sends 16 kHz PCM16 and this checkpoint's windows
are cut at 24 kHz.

Session affinity is unchanged: one in-process engine per replica, one uvicorn
worker, sessions never leave the replica that started them. `/v1/stream` and
the HTTP routes are still built by upstream and still there; only the routed
one changes.

A handshake applies to the turn that follows it: the session and the resampler
are built on the first audio frame, so `input_sample_rate` (and the sampling
knobs) must arrive before any audio, which is where a client sends them anyway.

Upstream is commit-pinned in config.yaml, and the names borrowed from it are
asserted below, so a bump that moves them fails loudly at boot instead of
serving a broken socket.
"""

from __future__ import annotations

import base64
import json
import logging
from fractions import Fraction

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from scipy import signal

import vllm_plugin.asr_streaming_server as upstream
from vllm_plugin.asr_streaming import chunk_segments, split_windows

logger = logging.getLogger("vibevoice.realtime")

# The route `docker_server.predict_endpoint` names, and the path the
# qwen_realtime / voxtral_realtime clients default to.
ROUTE = "/v1/realtime"

# b10-bench streaming protocols all send PCM16 mono at 16 kHz.
DEFAULT_INPUT_HZ = 16_000

_EMPTY = np.zeros(0, dtype=np.float32)

# Handshake keys we act on. Anything else is ignored and logged once, the same
# forward-compatible posture the qwen3-asr-streaming truss takes.
_APPLIED = {
    "whisper_params": {"prompt", "keyterms"},
    "streaming_params": {"enable_partial_transcripts", "input_sample_rate"},
    "vibevoice_params": {"context_info", "max_tokens", "temperature", "top_p",
                         "repetition_penalty"},
}
# Accepted and deliberately inert: this checkpoint has no language forcing, no
# word aligner, and no server-side VAD (turns are delimited by an explicit
# commit), so these carry no meaning here. Listed so they do not warn.
_INERT = {
    "whisper_params": {"audio_language", "show_word_timestamps",
                       "context_min_utter_s", "echo_guard", "min_decode_s"},
    "streaming_params": {"partial_transcript_interval_s",
                         "final_transcript_max_duration_s"},
    "vibevoice_params": set(),
}

for _name in ("create_app", "main", "TranscribeRequest", "_new_session"):
    assert hasattr(upstream, _name), (
        f"vllm_plugin.asr_streaming_server has no {_name!r} — upstream may have "
        f"changed; re-check the pinned VibeVoice commit in config.yaml"
    )


class Resampler:
    """Streaming rational resampler: exact ratio, no drift, no seam artifacts.

    `resample_poly` is stateless, so resampling each arriving frame on its own
    would put a filter transient at every frame boundary. This carries `PAD`
    input samples of context on each side (overlap-save) and only ever emits
    the middle, so the output is sample-identical to resampling the whole
    stream at once. Blocks are cut to a multiple of `down`, which keeps the
    output length exactly `n * up // down` and the input/output clocks locked.
    """

    # Input samples of context per side. resample_poly's default Kaiser FIR is
    # ~10*max(up, down) taps per phase, so 192 is a wide margin at 2:3.
    PAD = 192

    def __init__(self, src_hz: int, dst_hz: int):
        ratio = Fraction(int(dst_hz), int(src_hz))
        self.up, self.down = ratio.numerator, ratio.denominator
        self.passthrough = self.up == 1 and self.down == 1
        self._pending = _EMPTY  # input not yet consumed
        self._history = _EMPTY  # tail of consumed input, kept as left context

    def push(self, x: np.ndarray) -> np.ndarray:
        if self.passthrough:
            return x
        self._pending = np.concatenate([self._pending, x])
        n = len(self._pending) - self.PAD  # leave the right-side context behind
        n -= n % self.down
        return self._consume(n, self.PAD) if n > 0 else _EMPTY

    def flush(self) -> np.ndarray:
        """Drain the tail at end of turn (its right edge has no context left)."""
        if self.passthrough:
            return _EMPTY
        remainder = len(self._pending) % self.down
        if remainder:
            self._pending = np.concatenate(
                [self._pending, np.zeros(self.down - remainder, dtype=np.float32)])
        n = len(self._pending)
        out = self._consume(n, 0) if n > 0 else _EMPTY
        self._history = _EMPTY
        return out

    def _consume(self, n: int, right_pad: int) -> np.ndarray:
        history = self._history
        block = np.concatenate([history, self._pending[:n + right_pad]])
        y = signal.resample_poly(block, self.up, self.down)
        offset = len(history) * self.up // self.down
        out = y[offset:offset + n * self.up // self.down]
        keep = self.PAD - self.PAD % self.down  # multiple of `down`: offset stays exact
        self._history = np.array(self._pending[max(0, n - keep):n], dtype=np.float32)
        self._pending = self._pending[n:]
        return np.asarray(out, dtype=np.float32)


def pcm16_to_float32(raw: bytes) -> np.ndarray:
    """PCM16 little-endian mono -> float32 [-1, 1).

    "<i2" not np.int16: the wire is little-endian and the native dtype would
    reinterpret it on a big-endian host (upstream makes the same note about
    its float32 frames).
    """
    if len(raw) % 2:  # a split sample can only be a client framing bug
        raw = raw[:-1]
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def _apply_handshake(cfg: dict, obj: dict) -> None:
    """Fold a qwen_realtime-style handshake block into the session config."""
    for block, keys in _APPLIED.items():
        params = obj.get(block) or {}
        if not isinstance(params, dict):
            continue
        if unknown := (set(params) - keys - _INERT[block]):
            logger.warning("ignoring unknown %s keys: %s", block, sorted(unknown))
        for key in sorted(keys & set(params)):  # sorted: prompt/keyterms concat order
            value = params[key]
            if value is None:
                continue
            if key == "enable_partial_transcripts":
                cfg["partials"] = bool(value)
            elif key == "input_sample_rate":
                cfg["input_hz"] = int(value)
            elif key in ("prompt", "keyterms"):
                # Qwen's context channel is VibeVoice's hotword channel.
                text = ", ".join(value) if isinstance(value, (list, tuple)) else str(value)
                if text.strip():
                    prior = cfg["request"].get("context_info")
                    cfg["request"]["context_info"] = f"{prior}. {text}" if prior else text
            else:
                cfg["request"][key] = value


def mount_realtime(app: FastAPI) -> None:
    """Add the qwen_realtime translator to an app upstream already built."""

    @app.websocket(ROUTE)
    async def realtime(ws: WebSocket):  # noqa: C901 — one protocol loop, kept flat
        await ws.accept()
        geometry = app.state.geometry
        cfg = {"partials": True, "input_hz": DEFAULT_INPUT_HZ, "request": {}}

        session = None
        resampler = None
        buffer = _EMPTY
        texts: list[str] = []
        turn = 0

        def start_turn():
            nonlocal session, resampler, buffer, texts
            # A VibeVoice session is append-only, so a turn IS a session: a new
            # one starts here and the finished one's windows leave the cache.
            session = upstream._new_session(
                app.state.engine, upstream.TranscribeRequest(**cfg["request"]))
            resampler = Resampler(cfg["input_hz"], geometry.sample_rate)
            buffer = _EMPTY
            texts = []

        def transcription(is_final: bool, end_audio: bool = False) -> dict:
            """One qwen_realtime `transcription` message.

            Replace-style: every message re-renders the whole turn from all
            chunks so far, so a client displays the latest message and
            discards the previous one. `Content` is already free of the
            model's `Speaker N:` labels — upstream splits attribution out into
            `Speaker`, so the text scores against a plain reference and the
            speaker rides alongside rather than being thrown away.
            """
            message = {
                "type": "transcription",
                "is_final": is_final,
                "transcription_num": turn,
                "language_code": None,  # this checkpoint reports no language
                "segments": [
                    {"start_time": seg["Start"], "end_time": seg["End"],
                     "text": seg["Content"], "speaker": seg["Speaker"]}
                    for seg in chunk_segments(texts, geometry)
                ],
            }
            if end_audio:
                message["is_end_of_audio_flush"] = True
            return message

        try:
            while True:
                raw = await ws.receive_text()
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(data, dict):
                    continue
                mtype = data.get("type", "")

                if mtype == "input_audio_buffer.append":
                    audio_b64 = data.get("audio")
                    if not audio_b64:
                        continue
                    try:
                        pcm = base64.b64decode(audio_b64)
                    except Exception:  # noqa: BLE001 — drop a frame, not the stream
                        logger.warning("skipping malformed base64 audio frame")
                        continue
                    if session is None:
                        start_turn()
                    buffer = np.concatenate(
                        [buffer, resampler.push(pcm16_to_float32(pcm))])
                    # Upstream's own loop: emit whenever a full window is
                    # available, then advance by the text boundary so the
                    # lookahead overlap stays in the buffer.
                    while len(buffer) >= geometry.window_samples:
                        texts.append(await session.push(buffer[:geometry.window_samples]))
                        buffer = buffer[geometry.chunk_samples:]
                        if cfg["partials"]:
                            await ws.send_json(transcription(False))

                elif mtype == "input_audio_buffer.commit":
                    if session is not None:
                        buffer = np.concatenate([buffer, resampler.flush()])
                        for window in split_windows(buffer, geometry):
                            texts.append(await session.push(window))
                    await ws.send_json(transcription(True, end_audio=True))
                    session = None  # next append opens a fresh turn
                    turn += 1

                elif mtype == "transcription_session.update":
                    _apply_handshake(cfg, data.get("session") or {})

                elif any(k in data for k in
                         ("whisper_params", "streaming_params", "vibevoice_params",
                          "streaming_vad_config", "include_timing_info")):
                    _apply_handshake(cfg, data)

                # unknown message types are ignored, not fatal
        except WebSocketDisconnect:
            return
        except Exception as exc:  # noqa: BLE001 — report, then let the socket close
            logger.exception("realtime session failed")
            try:
                await ws.send_json({"type": "error", "error": str(exc)})
            except Exception:  # noqa: BLE001
                pass


_upstream_create_app = upstream.create_app


def create_app(config):
    """Upstream's app, plus the translator. Patched in so `main()` picks it up."""
    app = _upstream_create_app(config)
    mount_realtime(app)
    logger.info("mounted qwen_realtime translator on %s", ROUTE)
    return app


if __name__ == "__main__":
    upstream.create_app = create_app
    upstream.main()  # owns arg parsing, ServerConfig, logging and uvicorn
