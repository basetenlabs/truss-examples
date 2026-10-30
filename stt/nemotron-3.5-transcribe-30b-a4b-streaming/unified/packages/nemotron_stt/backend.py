"""Nemotron 3.5 Transcribe behind baseten-stt's ``STTBackend``.

The stream is the Qwen3-ASR unified preset's, renamed: buffering, the session clock, the
partial timer and cancellation are model-independent. The model gives one call -- PCM in,
text out, one full decode -- with no per-word or per-segment timings, so a turn becomes one segment spanning the audio the stream consumed.
The stream is the only thing that knows how much audio it has seen, so ``Segment`` timings
are measured from it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Protocol

from baseten_stt.backend.base import BackendStream, OnPartial, STTBackend
from baseten_stt.protocol.capabilities import (
    PCM16_16K_MONO,
    Capabilities,
    GrantedConfig,
    PartialStyle,
    SegmentationMode,
    VadConfig,
)
from baseten_stt.protocol.transcript import Segment

logger = logging.getLogger(__name__)

#: The protocol has no language field, so there is no choice to carry and "auto" is the only
#: value this backend asks for. Nemotron auto-detects; on 20 es/fr/de/pt clips it kept the
#: spoken language every time without a hint.
DECODE_LANGUAGE = "auto"

#: How much audio a partial re-decodes. Partials re-decode the unflushed span, so their cost
#: grows with the turn unless bounded; the final is always a full decode, so the bound costs
#: no accuracy.
DEFAULT_PARTIAL_WINDOW_S = 8.0


class _Engine(Protocol):
    """The model handle, kept to one member so the suite can drive a stub -- `the engine is an
    HTTP sidecar that needs the image's GPU stack."""

    async def transcribe(self, pcm: bytes, language: str = ...) -> tuple[str, str | None]:
        """One full decode. Returns the text and the language it was decoded as."""
        ...


class NemotronStream:
    """One session's buffer and decode state."""

    def __init__(
        self,
        engine: _Engine,
        config: GrantedConfig,
        on_partial: OnPartial | None,
        *,
        partial_window_s: float = DEFAULT_PARTIAL_WINDOW_S,
    ) -> None:
        self._engine = engine
        self._format = config.input_format
        self._buffer = bytearray()
        self._consumed_bytes = 0
        self._closed = False
        self._inflight: asyncio.Task[tuple[Segment, ...]] | None = None
        self._partial_decode: asyncio.Task[tuple[Segment, ...]] | None = None
        self._on_partial = on_partial
        self._partial_interval_s = config.partial_interval_s
        self._partial_window_s = partial_window_s
        self._partials: asyncio.Task[None] | None = None
        #: Bumped whenever the buffer is committed. A partial decode reads it before awaiting
        #: and drops its result if it has moved, since that audio belongs to a turn that has
        #: since been finalised or discarded and the connection would label the text with the
        #: next turn's id.
        self._turn_generation = 0
        #: The end of the last span a partial decoded, as (turn generation, stream byte). A
        #: client that pauses without flushing leaves the buffer as it was, and re-decoding
        #: it every tick would spend GPU on an answer the client already has.
        self._last_partial: tuple[int, int] | None = None

        #: Bytes to drop from the next append, to finish a sample whose first half went out
        #: with a discarded turn. See ``_commit``.
        self._skip_bytes = 0

    async def append(self, pcm: bytes) -> None:
        if self._closed:
            return
        # Started on first audio rather than at open, so a session that never speaks never
        # spends a decode, and the interval is measured from speech.
        if self._on_partial is not None and self._partials is None:
            self._partials = asyncio.create_task(self._emit_partials())
        if self._skip_bytes:
            # The other half of a sample that straddled a cleared turn. Dropping it keeps the
            # stream on sample boundaries without carrying discarded audio forward.
            dropped = min(self._skip_bytes, len(pcm))
            self._skip_bytes -= dropped
            self._consumed_bytes += dropped
            pcm = pcm[dropped:]
        self._buffer.extend(pcm)

    async def flush(self) -> tuple[Segment, ...]:
        # A partial decoding right now is about to describe a turn that no longer exists. The
        # generation check would discard its text anyway, so cancelling buys nothing in
        # correctness -- what it buys is the GPU, which the final decode is queued behind.
        self._cancel_partial_decode()
        start = self._format.duration_s(self._consumed_bytes)
        audio = self._commit()
        if not audio:
            return ()
        end = start + self._format.duration_s(len(audio))
        # Held locally as well as on self: ``clear`` may run while this is awaiting, and a
        # later flush may then install its own task. Clearing unconditionally in the finally
        # would drop that task's handle, leaving it uncancellable.
        task = asyncio.create_task(self._decode(audio, offset=start, until=end))
        self._inflight = task
        try:
            return await task
        finally:
            if self._inflight is task:
                self._inflight = None

    async def clear(self) -> None:
        self._cancel_inflight()
        self._cancel_partial_decode()
        self._commit(discard=True)

    async def close(self) -> None:
        self._closed = True
        await self._stop_partials()
        self._cancel_inflight()
        self._cancel_partial_decode()
        self._buffer.clear()
        self._skip_bytes = 0

    def _commit(self, *, discard: bool = False) -> bytes:
        """Retire the buffered audio in whole samples and advance the session clock over it.

        Whole samples only, because the server re-frames and a turn can therefore end
        mid-sample -- and ``np.frombuffer(..., dtype=np.int16)`` rejects an odd-length buffer
        outright. What happens to a straggler byte depends on why the turn ended:

        On a flush it stays buffered. It is real audio whose sample completes in the next
        turn, and dropping it would shift every later sample by one byte -- int16 read off a
        one-byte offset is noise, not quiet.

        On a discard it must not reach the next decode, or a cleared turn would leave one
        fabricated sample at the head of the next transcript. Alignment still has to hold, so
        the byte is dropped *and* its partner is dropped when it arrives.

        The clock advances over everything retired either way, discarded audio included: it
        did arrive, and rewinding would time every later turn as though it had not.
        """
        self._turn_generation += 1
        frame = self._format.bytes_per_frame
        whole = len(self._buffer) - len(self._buffer) % frame
        audio = bytes(self._buffer[:whole])
        del self._buffer[:whole]
        self._consumed_bytes += whole
        if discard and self._buffer:
            self._consumed_bytes += len(self._buffer)
            self._skip_bytes = frame - len(self._buffer)
            self._buffer.clear()
        return audio

    def _cancel_inflight(self) -> None:
        """Abandon the final decode.

        Cancelling the awaiting task closes the sidecar request's connection, and vLLM aborts
        a request whose client disconnected -- so this releases the GPU rather than merely
        dropping the answer.
        """
        task, self._inflight = self._inflight, None
        if task is not None and not task.done():
            task.cancel()

    def _cancel_partial_decode(self) -> None:
        """Abandon the partial decode, which is why it runs in a task of its own.

        Separate from ``_cancel_inflight`` rather than folded into one helper: a flush wants
        to stop the partial and must not touch a final, and keeping them apart means a change
        in concurrency behaviour cannot quietly turn one into the other.
        """
        task, self._partial_decode = self._partial_decode, None
        if task is not None and not task.done():
            task.cancel()

    async def _decode(self, audio: bytes, *, offset: float, until: float) -> tuple[Segment, ...]:
        text, language = await self._engine.transcribe(audio, DECODE_LANGUAGE)
        if not text.strip():
            # Silence, or a decode that produced only whitespace: an empty span says nothing
            # and still has to be rendered.
            return ()
        if language:
            logger.debug("decoded %.2fs as %s", until - offset, language)
        # One segment: the model returns an utterance, not a segmentation.
        return (Segment(text=text.strip(), start=offset, end=until),)

    async def _emit_partials(self) -> None:
        """Re-decode the trailing window on a timer, for as long as there is one.

        A partial is disposable. If a decode is already running the tick is skipped rather
        than queued -- the newer span supersedes the older answer, and a queue here is how a
        slow GPU becomes unbounded memory. A failed partial leaves the turn alone; the final
        is what the contract promises.
        """
        assert self._on_partial is not None
        # Not defaulted: the server grants an interval whenever it grants partials, and
        # quietly substituting one would emit at a rate the session was never told about.
        assert self._partial_interval_s is not None, "partials granted without an interval"
        interval = self._partial_interval_s
        frame = self._format.bytes_per_frame
        # A whole number of samples, so the window itself can never misalign the slice below.
        window_bytes = int(self._partial_window_s * self._format.sample_rate_hz) * frame
        while not self._closed:
            await asyncio.sleep(interval)
            if self._closed or self._inflight is not None or not self._buffer:
                continue
            # Both ends on sample boundaries. A trailing window taken from a raw length would
            # start mid-sample whenever the buffer held a straggler byte, and int16 read off a
            # one-byte offset is noise rather than the audio that arrived.
            whole = len(self._buffer) - len(self._buffer) % frame
            if not whole:
                continue
            snapshot = (self._turn_generation, self._consumed_bytes + whole)
            if snapshot == self._last_partial:
                continue
            head = max(0, whole - window_bytes)
            audio = bytes(self._buffer[head:whole])
            start = self._format.duration_s(self._consumed_bytes + head)
            until = start + self._format.duration_s(len(audio))
            generation = self._turn_generation
            # A task rather than a bare await, so ``clear`` can reach it. Awaited inline all
            # the same, which is what keeps one partial decode outstanding at a time.
            task = asyncio.create_task(self._decode(audio, offset=start, until=until))
            self._partial_decode = task
            try:
                segments = await task
            except asyncio.CancelledError:
                # Either the turn was discarded -- ``clear`` cancelled this decode and took
                # the handle with it -- or this loop is being torn down. Only the second is
                # ours to propagate, and it has to take the decode with it: cancelling the
                # awaiting coroutine does not cancel what it was awaiting.
                if self._partial_decode is task:
                    task.cancel()
                    raise
                continue
            except Exception:
                logger.exception("partial decode failed; the turn is unaffected")
                continue
            finally:
                if self._partial_decode is task:
                    self._partial_decode = None
            self._last_partial = snapshot
            if segments and not self._closed and generation == self._turn_generation:
                await self._on_partial(segments)

    async def _stop_partials(self) -> None:
        task, self._partials = self._partials, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


class NemotronBackend:
    """Nemotron 3.5 Transcribe, one engine per replica, shared across sessions.

    The sidecar vLLM batches concurrent requests continuously, so sessions share the engine
    rather than queueing behind each other.
    """

    def __init__(
        self, engine: _Engine, *, partial_window_s: float = DEFAULT_PARTIAL_WINDOW_S
    ) -> None:
        self._engine = engine
        self._partial_window_s = partial_window_s

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            input_format=PCM16_16K_MONO,
            # Both, because turn control is the server's either way: this backend never
            # decides a boundary, so server_vad is the pipeline's VAD, not the model's.
            segmentation_modes=(SegmentationMode.CLIENT_FLUSH, SegmentationMode.SERVER_VAD),
            partials=True,
            # Each partial is a fresh decode of the trailing window and supersedes the last.
            partial_style=PartialStyle.REPLACE,
            # 0.5 s of trailing silence ends a turn, not the server's 0.2 s. Measured on 100
            # pipecat clips with the server default: server_vad scored 6.6% naive WER against
            # 2.2% for client_flush on the same clips, because 0.2 s splits sentences at
            # breaths. Qwen3-ASR's preset measured the same effect and settled on these values;
            # the other three restate the server's defaults.
            vad=VadConfig(silence_s=0.5, speech_start_s=0.2, threshold=0.7, min_volume=0.0),
            #
            # The replica's admission ceiling. Above predict_concurrency (32), so connection
            # churn does not refuse a session the platform has already routed.
            max_sessions_per_replica=48,
        )

    async def open(
        self, config: GrantedConfig, *, on_partial: OnPartial | None = None
    ) -> BackendStream:
        return NemotronStream(
            self._engine,
            config,
            on_partial if config.partials else None,
            partial_window_s=self._partial_window_s,
        )

    async def close(self) -> None:
        # The engine outlives this object: the truss loaded it and the truss owns it. Closing
        # it here would take the model down while the replica is still serving.
        return None


# Conformance is checked statically -- an annotated assignment is what catches a signature
# drifting out of shape, since isinstance against a Protocol only checks attributes exist.
_backend_conforms: type[STTBackend] = NemotronBackend
_stream_conforms: type[BackendStream] = NemotronStream
