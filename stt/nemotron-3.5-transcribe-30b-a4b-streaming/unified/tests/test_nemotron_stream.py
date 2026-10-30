"""Frame alignment and decode cancellation in ``NemotronStream``.

Not run by this repository's CI, which has no Python test job -- run by hand with

    uv run --with pytest --with pytest-asyncio --with baseten-stt \
        pytest stt/nemotron-3.5-transcribe-30b-a4b-streaming/unified/tests

The engine is a stub throughout. ``vllm`` is not installable outside the truss image and an
engine load needs a GPU, so the real thing cannot be reached from a test at all -- which is
what ``_Engine`` being a one-member Protocol is for.
"""

from __future__ import annotations

import asyncio

import pytest
from nemotron_stt.backend import NemotronBackend, NemotronStream

from baseten_stt.protocol.capabilities import (
    PCM16_16K_MONO,
    GrantedConfig,
    GrantedContext,
    SegmentationMode,
)

pytestmark = pytest.mark.asyncio

FRAME = PCM16_16K_MONO.bytes_per_frame

#: Long enough that a loaded machine does not fail a passing case, short enough that a real
#: hang is reported rather than waited on.
WAIT_S = 1.0


def config(
    *,
    partials: bool = False,
    interval: float | None = None,
    language: str | None = None,
    context: GrantedContext | None = None,
) -> GrantedConfig:
    return GrantedConfig(
        input_format=PCM16_16K_MONO,
        segmentation_mode=SegmentationMode.CLIENT_FLUSH,
        partials=partials,
        partial_interval_s=interval,
        language=language,
        context=context,
    )


class RecordingEngine:
    """Answers every decode, and remembers the byte counts it was handed."""

    def __init__(self, text: str = "hello") -> None:
        self.text = text
        self.seen: list[int] = []
        #: The (language, context) of every decode, in order.
        self.asked: list[tuple[str, GrantedContext | None]] = []

    async def transcribe(
        self, pcm: bytes, language: str = "auto", context: GrantedContext | None = None
    ) -> tuple[str, str | None]:
        self.seen.append(len(pcm))
        self.asked.append((language, context))
        # The failure this guards: numpy rejects a buffer that is not a whole number of
        # samples, so an odd length reaching a real engine is a crashed turn.
        assert len(pcm) % FRAME == 0, f"odd-length PCM reached the engine: {len(pcm)}"
        return self.text, None


class BlockingEngine:
    """Never answers until released, so a decode can be observed mid-flight."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        #: Set from the decode's own cancellation handler. ``task.cancel()`` merely schedules
        #: that handler, so a test that asserted straight after cancelling would read the
        #: state one or more ticks too early -- awaiting this is what makes the check about
        #: the abort having happened rather than about how long a sleep was guessed to be.
        self.aborted = asyncio.Event()
        self.completed = 0

    async def transcribe(
        self, pcm: bytes, language: str = "auto", context: GrantedContext | None = None
    ) -> tuple[str, str | None]:
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.aborted.set()
            raise
        self.completed += 1
        return "text", None


# --- odd-byte PCM -----------------------------------------------------------------------


async def test_an_odd_flush_decodes_whole_samples_only() -> None:
    engine = RecordingEngine()
    stream = NemotronStream(engine, config(), None)

    await stream.append(b"\x01" * 101)
    segments = await stream.flush()

    assert engine.seen == [100]
    assert segments and segments[0].end == pytest.approx(100 / (FRAME * 16000))


async def test_a_straggler_byte_survives_a_flush_and_completes_next_turn() -> None:
    """Dropping it would shift every later sample by one byte, which reads as noise."""
    engine = RecordingEngine()
    stream = NemotronStream(engine, config(), None)

    await stream.append(b"\x01" * 3)
    await stream.flush()
    await stream.append(b"\x02" * 3)
    await stream.flush()

    # 3 bytes -> one sample decoded, one byte held; +3 -> four bytes, two samples.
    assert engine.seen == [2, 4]


async def test_a_lone_odd_byte_decodes_nothing_rather_than_crashing() -> None:
    engine = RecordingEngine()
    stream = NemotronStream(engine, config(), None)

    await stream.append(b"\x01")

    assert await stream.flush() == ()
    assert engine.seen == []


async def test_a_discarded_straggler_takes_its_partner_with_it() -> None:
    """Otherwise a cleared turn leaves one fabricated sample at the head of the next."""
    engine = RecordingEngine()
    stream = NemotronStream(engine, config(), None)

    await stream.append(b"\x01" * 3)
    await stream.clear()
    await stream.append(b"\x02" * 4)
    await stream.flush()

    # The clear kept 1 byte of a sample and dropped it; the next append's first byte is
    # dropped to restore alignment, leaving 3 bytes buffered -> one whole sample decoded.
    assert engine.seen == [2]


async def test_the_clock_advances_over_discarded_audio() -> None:
    engine = RecordingEngine()
    stream = NemotronStream(engine, config(), None)

    await stream.append(b"\x01" * 3)
    await stream.clear()
    await stream.append(b"\x02" * 4)
    segments = await stream.flush()

    # 3 discarded + 1 dropped partner = 4 bytes retired before this turn opens.
    assert segments and segments[0].start == pytest.approx(4 / (FRAME * 16000))


@pytest.mark.parametrize("sizes", [(1, 1), (3, 1, 1), (1, 2, 1, 5), (7,)])
async def test_no_split_of_a_sample_across_appends_reaches_the_engine_odd(
    sizes: tuple[int, ...],
) -> None:
    engine = RecordingEngine()
    stream = NemotronStream(engine, config(), None)

    for size in sizes:
        await stream.append(b"\x01" * size)
    await stream.flush()

    assert all(seen % FRAME == 0 for seen in engine.seen)


# --- cancellation -----------------------------------------------------------------------


async def test_a_flush_cancels_the_obsolete_partial_before_deciding_the_final() -> None:
    """The partial's text is generation-filtered anyway; what cancelling saves is the GPU
    the final decode is queued behind."""
    engine = BlockingEngine()
    stream = NemotronStream(engine, config(partials=True, interval=0.01), _swallow)

    await stream.append(b"\x01" * 3200)
    await asyncio.wait_for(engine.started.wait(), timeout=WAIT_S)

    flush = asyncio.create_task(stream.flush())
    await asyncio.wait_for(engine.aborted.wait(), timeout=WAIT_S)

    assert engine.completed == 0, "the final decode began behind the obsolete partial"

    engine.release.set()
    await asyncio.wait_for(flush, timeout=WAIT_S)
    await stream.close()


async def test_a_clear_cancels_the_partial_decode() -> None:
    engine = BlockingEngine()
    stream = NemotronStream(engine, config(partials=True, interval=0.01), _swallow)

    await stream.append(b"\x01" * 3200)
    await asyncio.wait_for(engine.started.wait(), timeout=WAIT_S)
    await stream.clear()
    await asyncio.wait_for(engine.aborted.wait(), timeout=WAIT_S)

    engine.release.set()
    await stream.close()


async def test_a_cancelled_partial_leaves_the_loop_running() -> None:
    """A discarded turn must not end partials for the rest of the session."""
    engine = BlockingEngine()
    stream = NemotronStream(engine, config(partials=True, interval=0.01), _swallow)

    await stream.append(b"\x01" * 3200)
    await asyncio.wait_for(engine.started.wait(), timeout=WAIT_S)
    await stream.clear()
    await asyncio.wait_for(engine.aborted.wait(), timeout=WAIT_S)

    engine.started.clear()
    await stream.append(b"\x02" * 3200)
    # The assertion is that this does not time out: a discarded turn must not end partials
    # for the rest of the session.
    await asyncio.wait_for(engine.started.wait(), timeout=WAIT_S)

    engine.release.set()
    await stream.close()


async def test_a_paused_client_is_not_re_decoded_every_tick() -> None:
    """Audio that has not changed since the last partial is not sent to the GPU again."""
    engine = RecordingEngine()
    stream = NemotronStream(engine, config(partials=True, interval=0.01), _swallow)

    await stream.append(b"\x01" * 3200)
    await asyncio.sleep(0.2)
    assert engine.seen == [3200]

    await stream.append(b"\x02" * 3200)
    await asyncio.sleep(0.2)
    assert engine.seen == [3200, 6400]
    await stream.close()


async def test_partials_granted_without_an_interval_is_a_bug_not_a_default() -> None:
    """Substituting a rate would emit at one the session was never told about."""
    engine = RecordingEngine()
    stream = NemotronStream(engine, config(partials=True, interval=None), _swallow)

    await stream.append(b"\x01" * 3200)
    with pytest.raises(AssertionError, match="without an interval"):
        await asyncio.wait_for(stream._partials, timeout=WAIT_S)


# --- language and context ---------------------------------------------------------------


async def test_a_session_with_no_language_decodes_with_detection() -> None:
    """A grant without a language -- a server that negotiates none -- decodes as before."""
    engine = RecordingEngine()
    stream = NemotronStream(engine, config(), None)

    await stream.append(b"\x01" * 3200)
    await stream.flush()
    assert engine.asked == [("auto", None)]


async def test_every_decode_carries_the_granted_language_and_context() -> None:
    """Partials and finals alike: a partial in another language would be replaced by a final
    that disagrees with it."""
    context = GrantedContext(text="A retail support call.", keyterms=("W2702727",))
    engine = RecordingEngine()
    stream = NemotronStream(
        engine, config(partials=True, interval=0.01, language="en", context=context), _swallow
    )

    await stream.append(b"\x01" * 3200)
    await asyncio.sleep(0.1)
    await stream.flush()
    await stream.close()
    assert len(engine.asked) >= 2
    assert set(engine.asked) == {("en", context)}


async def test_capabilities_declare_the_cards_languages_and_a_context_slot() -> None:
    caps = NemotronBackend(RecordingEngine()).capabilities
    assert len(caps.languages) == 41 and "en" in caps.languages and "auto" in caps.languages
    assert caps.default_language == "auto"
    assert caps.context is not None and caps.context.text and caps.context.keyterms
    # Declared as BCP-47 codes, so a client is told exactly what the backend declared.
    assert caps.on_the_wire().languages == caps.languages


async def _swallow(segments: object) -> None:
    return None
