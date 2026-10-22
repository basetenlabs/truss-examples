"""PIPELINE_OVERLAP in model/model.py: the encode/decode pipeline and its dispatch.

Not run by this repository's CI, which has no Python test job -- run by hand with

    uv run --with pytest --with numpy --with prometheus-client pytest stt/parakeet-tdt-0.6b-v3/batch/tests

CPU only, with torch.cuda mocked: these cover the decode thread's ordering, the
one-decode-at-a-time and depth-1 handoff bounds, per-batch error delivery, drain,
shutdown, and which batches the batch worker overlaps or drains for. Real streams,
events and allocator lifetimes need a GPU; the deployment's startup parity check
decodes one batch serially and twice through the pipeline and fails load() on any
transcript difference.
"""

from __future__ import annotations

import contextlib
import importlib.util
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "model.py"
SR = 16_000
CONFIG_KEYS = (
    "PIPELINE_OVERLAP",
    "ENCODER_CUDA_GRAPHS",
    "PRECISION_MODE",
    "INFERENCE_DTYPE",
    "TIMING_LOG_BATCHES",
    "GRAPH_PARITY_BATCHES",
)


def _load(monkeypatch, **env):
    from prometheus_client import REGISTRY

    for collector, names in list(REGISTRY._collector_to_names.items()):
        if any(name.startswith("parakeet_") for name in names):
            REGISTRY.unregister(collector)
    for key in CONFIG_KEYS:
        monkeypatch.delenv(key, raising=False)
    base = {
        "MAX_BATCH_SIZE": "16",
        "BATCH_WINDOW_MS": "5",
        "BATCH_BUCKET_SECONDS": "2,4,8",
        "PARAKEET_DIRECT_FORWARD": "true",
    }
    for key, value in {**base, **env}.items():
        monkeypatch.setenv(key, value)
    spec = importlib.util.spec_from_file_location("parakeet_tdt_overlap", MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- configuration ----------------------------------------------------------------


def test_overlap_is_off_by_default(monkeypatch):
    assert _load(monkeypatch).PIPELINE_OVERLAP is False


def test_overlap_with_encoder_graphs(monkeypatch):
    mm = _load(monkeypatch, PIPELINE_OVERLAP="true", ENCODER_CUDA_GRAPHS="true")
    assert mm.PIPELINE_OVERLAP is True


@pytest.mark.parametrize(
    "env",
    [
        {"PIPELINE_OVERLAP": "true"},  # needs ENCODER_CUDA_GRAPHS
        {"PIPELINE_OVERLAP": "maybe", "ENCODER_CUDA_GRAPHS": "true"},
    ],
)
def test_invalid_overlap_config_is_rejected(monkeypatch, env):
    with pytest.raises(ValueError, match="PIPELINE_OVERLAP"):
        _load(monkeypatch, **env)


def test_config_yaml_ships_overlap_on():
    yaml = pytest.importorskip("yaml")
    config = yaml.safe_load((MODEL_PATH.parents[1] / "config.yaml").read_text())
    env = config["environment_variables"]
    assert env["PIPELINE_OVERLAP"] == "true"
    assert env["ENCODER_CUDA_GRAPHS"] == "true"


# --- OverlapPipeline --------------------------------------------------------------


@pytest.fixture
def mm(monkeypatch):
    return _load(monkeypatch)


class Collector:
    def __init__(self):
        self.results = {}
        self.calls = []
        self.lock = threading.Lock()

    def on_done(self, key):
        def deliver(result):
            with self.lock:
                self.calls.append(key)
                self.results[key] = result

        return deliver


def _pipeline(mm, decode_fn, timeout=5.0):
    syncs = []
    pipeline = mm.OverlapPipeline(decode_fn, lambda: syncs.append(time.monotonic()), timeout)
    return pipeline, syncs


def test_batches_decode_in_submit_order_one_at_a_time(mm):
    active, peak, order = [0], [0], []
    lock = threading.Lock()

    def decode(handoff):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.002)
        order.append(handoff)
        with lock:
            active[0] -= 1
        return [f"out {handoff}"]

    pipeline, syncs = _pipeline(mm, decode)
    collector = Collector()
    for index in range(20):
        pipeline.submit(index, collector.on_done(index))
    pipeline.drain()
    assert order == list(range(20))
    assert collector.calls == list(range(20))
    assert collector.results[7] == ["out 7"]
    assert peak[0] == 1
    assert len(syncs) == 1
    pipeline.close()


def test_submit_blocks_while_one_batch_already_waits(mm):
    release = threading.Event()
    started = threading.Event()

    def decode(handoff):
        started.set()
        release.wait(5)
        return [handoff]

    pipeline, _ = _pipeline(mm, decode)
    collector = Collector()
    pipeline.submit("a", collector.on_done("a"))  # decoding
    assert started.wait(5)
    pipeline.submit("b", collector.on_done("b"))  # waits in the depth-1 queue
    third = threading.Thread(target=pipeline.submit, args=("c", collector.on_done("c")))
    third.start()
    third.join(0.2)
    assert third.is_alive(), "a third batch must wait for the decode stage"
    release.set()
    third.join(5)
    pipeline.drain()
    assert collector.calls == ["a", "b", "c"]
    pipeline.close()


def test_pipeline_decode_error_reaches_only_its_batch(mm):
    def decode(handoff):
        if handoff == 1:
            raise ValueError("bad batch")
        return [handoff]

    pipeline, _ = _pipeline(mm, decode)
    collector = Collector()
    for index in range(3):
        pipeline.submit(index, collector.on_done(index))
    pipeline.drain()
    assert collector.results[0] == [0]
    assert isinstance(collector.results[1], ValueError)
    assert collector.results[2] == [2]
    # A failed decode does not stop the pipeline.
    pipeline.submit(3, collector.on_done(3))
    pipeline.drain()
    assert collector.results[3] == [3]
    pipeline.close()


def test_failed_delivery_stops_accepting_batches(mm):
    pipeline, _ = _pipeline(mm, lambda handoff: [handoff])

    def explode(_result):
        raise RuntimeError("delivery bug")

    pipeline.submit(0, explode)
    pipeline.drain()
    with pytest.raises(RuntimeError, match="failed"):
        pipeline.submit(1, lambda _result: None)
    pipeline.close()


def test_drain_waits_for_every_batch_then_syncs(mm):
    release = threading.Event()

    def decode(handoff):
        release.wait(5)
        return [handoff]

    pipeline, syncs = _pipeline(mm, decode)
    collector = Collector()
    pipeline.submit(0, collector.on_done(0))
    pipeline.submit(1, collector.on_done(1))
    drained = threading.Event()
    drainer = threading.Thread(target=lambda: (pipeline.drain(), drained.set()))
    drainer.start()
    assert not drained.wait(0.2)
    assert syncs == []
    release.set()
    assert drained.wait(5)
    assert collector.calls == [0, 1]
    assert len(syncs) == 1
    pipeline.close()


def test_drain_times_out_on_a_stuck_decode(mm):
    release = threading.Event()
    pipeline, syncs = _pipeline(mm, lambda h: release.wait(5) and [h], timeout=0.2)
    pipeline.submit(0, lambda _result: None)
    with pytest.raises(TimeoutError, match="still has 1 batches"):
        pipeline.drain()
    assert syncs == []
    release.set()
    pipeline.close()


def test_exclusive_drains_before_and_syncs_after_even_on_error(mm):
    events = []
    pipeline = mm.OverlapPipeline(
        lambda handoff: events.append(("decode", handoff)) or [handoff],
        lambda: events.append(("sync",)),
        5.0,
    )
    pipeline.submit(0, lambda _result: None)
    with pytest.raises(ValueError):
        with pipeline.exclusive():
            events.append(("serial",))
            raise ValueError("serial batch failed")
    assert events == [("decode", 0), ("sync",), ("serial",), ("sync",)]
    pipeline.close()


def test_close_decodes_handed_over_batches_and_refuses_new_ones(mm):
    pipeline, _ = _pipeline(mm, lambda handoff: [handoff])
    collector = Collector()
    pipeline.submit(0, collector.on_done(0))
    pipeline.submit(1, collector.on_done(1))
    pipeline.close()
    assert collector.results == {0: [0], 1: [1]}
    assert not pipeline._thread.is_alive()
    with pytest.raises(RuntimeError, match="shut down"):
        pipeline.submit(2, collector.on_done(2))
    pipeline.close()  # idempotent


def test_close_fails_batches_a_hung_decode_never_reached(mm):
    release = threading.Event()
    started = threading.Event()

    def decode(handoff):
        started.set()
        release.wait(5)
        return [handoff]

    pipeline, _ = _pipeline(mm, decode, timeout=0.3)
    collector = Collector()
    pipeline.submit(0, collector.on_done(0))
    assert started.wait(5)
    pipeline.submit(1, collector.on_done(1))
    pipeline.close()
    assert isinstance(collector.results[1], RuntimeError)
    release.set()
    deadline = time.monotonic() + 5
    while 0 not in collector.results and time.monotonic() < deadline:
        time.sleep(0.01)
    assert collector.results[0] == [0]
    assert collector.calls.count(0) == 1 and collector.calls.count(1) == 1


# --- Model dispatch with mocked CUDA ----------------------------------------------


class FakeEvent:
    clock = [0.0]

    def __init__(self, enable_timing=False):
        self.enable_timing = enable_timing
        self.stream = None
        self.at = None

    def record(self, stream=None):
        self.stream = stream if stream is not None else FakeTorch.current.get(threading.get_ident())
        FakeEvent.clock[0] += 1.0
        self.at = FakeEvent.clock[0]

    def synchronize(self):
        pass

    def elapsed_time(self, other):
        assert self.enable_timing and other.enable_timing
        return other.at - self.at


class FakeStream:
    def __init__(self, name="side"):
        self.name = name
        self.waited = []

    def wait_event(self, event):
        self.waited.append(event)


class FakeTorch(types.ModuleType):
    current = {}

    def __init__(self):
        super().__init__("torch")
        self.inference_mode = contextlib.nullcontext
        self.autocast = lambda **_kwargs: contextlib.nullcontext()
        self.syncs = 0
        default = FakeStream("default")
        torch = self

        @contextlib.contextmanager
        def stream(s):
            thread = threading.get_ident()
            previous = FakeTorch.current.get(thread, default)
            FakeTorch.current[thread] = s
            try:
                yield
            finally:
                FakeTorch.current[thread] = previous

        def synchronize():
            torch.syncs += 1

        names = iter(("encode", "decode"))
        self.cuda = types.SimpleNamespace(
            Event=FakeEvent,
            Stream=lambda: FakeStream(next(names)),
            stream=stream,
            synchronize=synchronize,
        )
        self.backends = types.SimpleNamespace(cudnn=types.SimpleNamespace(enabled=False))
        FakeTorch.current.clear()


class FakeTensor:
    def __init__(self, batch, frames, made_on):
        self.shape = (batch, 1024, frames)
        self.made_on = made_on
        self.recorded_on = []

    def record_stream(self, stream):
        self.recorded_on.append(stream.name)


class FakeGraphs:
    batch_sizes = (1, 2, 4, 8, 16)
    max_padded = 30 * SR
    disabled_reason = None

    def __init__(self, fail_encode=()):
        self.runs = []
        self.fail_encode = set(fail_encode)

    def replay_key(self, waveforms):
        longest = max(w.shape[0] for w in waveforms)
        return (len(waveforms), longest) if longest <= self.max_padded else None

    def run(self, waveforms, staging=None):
        stream = FakeTorch.current[threading.get_ident()]
        tag = _tag(waveforms)
        self.runs.append((tag, stream.name, staging))
        if tag in self.fail_encode:
            raise RuntimeError(f"encode failed for {tag}")
        return FakeTensor(len(waveforms), 10, stream.name), FakeTensor(len(waveforms), 1, stream.name)


def _clip(tag, seconds=1.0):
    clip = np.zeros(round(seconds * SR), dtype=np.float32)
    clip[0] = tag
    return clip


def _tag(waveforms):
    return int(waveforms[0][0])


class FakeDecoding:
    def __init__(self, log, fail=()):
        self.log = log
        self.fail = set(fail)
        self.tensors = []

    def rnnt_decoder_predictions_tensor(self, encoder_output, encoded_lengths, return_hypotheses):
        stream = FakeTorch.current.get(threading.get_ident())
        self.tensors.append(encoder_output)
        tag = int(encoder_output.tag)
        self.log.append(("decode", tag, stream.name if stream else "default"))
        if tag in self.fail:
            raise ValueError(f"decode failed for {tag}")
        return [
            types.SimpleNamespace(text=f"text {tag}.{row}")
            for row in range(encoder_output.shape[0])
        ]


@pytest.fixture
def overlap(monkeypatch):
    torch = FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    mm = _load(monkeypatch, PIPELINE_OVERLAP="true", ENCODER_CUDA_GRAPHS="true")
    monkeypatch.setattr(mm, "PinnedStaging", lambda rows, samples: ("staging", rows, samples))
    log = []

    model = mm.Model()
    model._model = types.SimpleNamespace(decoding=FakeDecoding(log))
    model._serving_decoding = model._model.decoding
    model._graphs = FakeGraphs()

    # Tag encoder outputs with the batch they came from, for the fake decoder.
    original_run = model._graphs.run

    def tagged_run(waveforms, staging=None):
        result = original_run(waveforms, staging)
        for tensor in result:
            tensor.tag = _tag(waveforms)
        return result

    model._graphs.run = tagged_run
    serial = []

    def infer_waveforms(waveforms, *, with_timestamps):
        log.append(("serial", _tag(waveforms), with_timestamps))
        serial.append(_tag(waveforms))
        return [types.SimpleNamespace(text=f"text {_tag(waveforms)}.{i}") for i in range(len(waveforms))]

    model._infer_waveforms = infer_waveforms
    model._start_pipeline()
    yield types.SimpleNamespace(mm=mm, model=model, torch=torch, log=log, serial=serial)
    model.shutdown()


def _jobs(tag, rows=2, seconds=1.0, timestamps=False):
    return [
        {
            "waveform": _clip(tag, seconds),
            "timestamps": timestamps,
            "selected_at": time.monotonic(),
            "done": threading.Event(),
        }
        for _ in range(rows)
    ]


def _wait(jobs):
    for job in jobs:
        assert job["done"].wait(5)


def test_start_pipeline_creates_streams_staging_and_thread(overlap):
    model = overlap.model
    assert model._encode_stream.name == "encode"
    assert model._decode_stream.name == "decode"
    assert model._staging == ("staging", 16, 30 * SR)
    assert model._pipeline._thread.is_alive()


def test_overlapped_batches_encode_and_decode_on_their_streams(overlap):
    model = overlap.model
    batches = [_jobs(tag) for tag in (1, 2, 3)]
    for jobs in batches:
        model._run_jobs(jobs)
    for jobs in batches:
        _wait(jobs)
    assert [job["output"].text for job in batches[1]] == ["text 2.0", "text 2.1"]
    assert [(tag, stream) for tag, stream, _ in model._graphs.runs] == [
        (1, "encode"), (2, "encode"), (3, "encode")
    ]
    assert all(staging == model._staging for _, _, staging in model._graphs.runs)
    assert [entry for entry in overlap.log if entry[0] == "decode"] == [
        ("decode", 1, "decode"), ("decode", 2, "decode"), ("decode", 3, "decode")
    ]
    # The decode stream waited on each batch's encoder event, and the encoder output
    # is tied to the decode stream for the allocator.
    assert len(model._decode_stream.waited) == 3
    assert all(event.stream.name == "encode" for event in model._decode_stream.waited)
    tensor = model._model.decoding.tensors[0]
    assert tensor.made_on == "encode" and tensor.recorded_on == ["decode"]
    assert overlap.serial == []


@pytest.mark.parametrize(
    "case",
    ["timestamps", "eager_length", "decoding_rebuild", "graph_parity"],
)
def test_non_overlappable_batches_drain_then_run_serially(overlap, monkeypatch, case):
    model, torch = overlap.model, overlap.torch
    first = _jobs(1)
    model._run_jobs(first)
    if case == "timestamps":
        second = _jobs(2, timestamps=True)
    elif case == "eager_length":
        second = _jobs(2, seconds=31)
    elif case == "decoding_rebuild":
        second = _jobs(2)
        model._model.decoding = FakeDecoding(overlap.log)
    else:
        second = _jobs(2)
        monkeypatch.setattr(overlap.mm, "GRAPH_PARITY_BATCHES", 1)
        monkeypatch.setitem(overlap.mm._TIMING_STATE, "serving", True)
    syncs_before = torch.syncs
    model._run_jobs(second)
    _wait(first + second)
    # Batch 1's decode finished before batch 2 started, then the device synced.
    assert overlap.log.index(("decode", 1, "decode")) < overlap.log.index(
        ("serial", 2, case == "timestamps")
    )
    # One sync after the drain, one after the serial batch.
    assert torch.syncs == syncs_before + 2
    assert model._pipeline.drains == 1
    assert second[0]["output"].text == "text 2.0"


def test_model_encode_error_reaches_only_its_batch(overlap):
    model = overlap.model
    model._graphs.fail_encode.add(2)
    batches = [_jobs(tag) for tag in (1, 2, 3)]
    for jobs in batches:
        model._run_jobs(jobs)
    for jobs in batches:
        _wait(jobs)
    assert "error" not in batches[0][0] and "error" not in batches[2][0]
    assert all("encode failed for 2" in str(job["error"]) for job in batches[1])


def test_model_decode_error_reaches_only_its_batch(overlap):
    model = overlap.model
    model._model.decoding.fail.add(2)
    batches = [_jobs(tag) for tag in (1, 2, 3)]
    for jobs in batches:
        model._run_jobs(jobs)
    for jobs in batches:
        _wait(jobs)
    assert batches[0][1]["output"].text == "text 1.1"
    assert all(isinstance(job["error"], ValueError) for job in batches[1])
    assert batches[2][0]["output"].text == "text 3.0"


def test_batches_after_shutdown_fail_instead_of_hanging(overlap):
    model = overlap.model
    model.shutdown()
    jobs = _jobs(1)
    model._run_jobs(jobs)
    _wait(jobs)
    assert all("shut down" in str(job["error"]) for job in jobs)


def test_timed_overlapped_batch_uses_events_not_device_sync(overlap, monkeypatch, caplog):
    mm, model, torch = overlap.mm, overlap.model, overlap.torch
    monkeypatch.setattr(mm, "TIMING_LOG_BATCHES", 5)
    monkeypatch.setitem(mm._TIMING_STATE, "serving", True)
    monkeypatch.setitem(mm._TIMING_STATE, "batches", 0)
    syncs_before = torch.syncs
    jobs = _jobs(1)
    with caplog.at_level("INFO", logger=mm.LOGGER.name):
        model._run_jobs(jobs)
        _wait(jobs)
        model._pipeline.drain()
    assert torch.syncs == syncs_before + 1  # the drain only
    line = next(r.getMessage() for r in caplog.records if "batch timing" in r.getMessage())
    for field in (
        "batch=2", "encoder_path=graph", "encoder_frames=10", "encoder_ms=",
        "decoder_ms=", "pipeline=overlap", "encoder_start_ms=", "decoder_start_ms=",
        "handoff_wait_ms=",
    ):
        assert field in line, field
    assert mm._TIMING_STATE["batches"] == 1


def test_startup_parity_check_passes_on_identical_transcripts(overlap):
    model = overlap.model
    clip = _clip(4, seconds=6)
    model._infer_waveforms = lambda waveforms, *, with_timestamps: [
        types.SimpleNamespace(text=f"text 4.{row}") for row in range(len(waveforms))
    ]
    model._check_pipeline_parity(clip)
    decodes = [entry for entry in overlap.log if entry[0] == "decode"]
    assert decodes == [("decode", 4, "decode"), ("decode", 4, "decode")]
    assert len(model._model.decoding.tensors) == 2
    assert model._model.decoding.tensors[0].shape[0] == 4


def test_startup_parity_check_fails_on_any_difference(overlap):
    model = overlap.model
    model._infer_waveforms = lambda waveforms, *, with_timestamps: [
        types.SimpleNamespace(text="serial text") for _ in waveforms
    ]
    with pytest.raises(RuntimeError, match="pipeline parity check failed"):
        model._check_pipeline_parity(_clip(4, seconds=6))


def test_startup_parity_check_fails_when_a_pass_errors(overlap):
    model = overlap.model
    model._model.decoding.fail.add(4)
    model._infer_waveforms = lambda waveforms, *, with_timestamps: [
        types.SimpleNamespace(text=f"text 4.{row}") for row in range(len(waveforms))
    ]
    with pytest.raises(RuntimeError, match="overlapped pass 0 failed"):
        model._check_pipeline_parity(_clip(4, seconds=6))


# --- pinned staging ---------------------------------------------------------------


def test_stack_waveforms_unchanged_by_fill_refactor(mm):
    signals, lengths = mm._stack_waveforms(
        [np.ones(3, dtype=np.float32), np.ones(5, dtype=np.float32)], 6, rows=4
    )
    assert signals.tolist() == [
        [1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0], [0] * 6, [0] * 6
    ]
    assert lengths.tolist() == [3, 5, 6, 6]
    with pytest.raises(ValueError, match="do not fit"):
        mm._stack_waveforms([np.ones(3, dtype=np.float32)] * 3, 6, rows=2)


def test_pinned_staging_waits_for_a_slot_before_reusing_it(mm):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("pinned memory needs CUDA")
    staging = mm.PinnedStaging(rows=2, samples=8)
    slot, signal, lengths = staging.stage([np.ones(3, dtype=np.float32)], 2, 4)
    assert slot == 0 and signal.shape == (2, 4) and lengths.tolist() == [3, 4]
    staging.copied(slot)
    assert staging.stage([np.ones(2, dtype=np.float32)], 1, 2)[0] == 1
    assert staging.stage([np.ones(2, dtype=np.float32)], 1, 2)[0] == 0
