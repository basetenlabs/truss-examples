"""Fluid Speech additions to model/model.py: the block/hook load check, the CUDA health
probe and fail-fast after any CUDA error, and the recent-batch log.

Not run by this repository's CI, which has no Python test job -- run by hand with

    uv run --with pytest --with numpy --with soundfile --with prometheus-client --with torch \
        pytest stt/fluid-speech-pro/batch-bf16/tests

CPU only. The block test needs torch (skipped without it) and builds a toy encoder;
the real checkpoint, cast and CUDA-graph capture are exercised by the deployment's
startup checks.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import tarfile
import threading
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model" / "model.py"
LOADER_PATH = ROOT / "model" / "fluid_speech.py"
# sha256 of fluid_speech.py at FluidVoice-Labs/fluid-speech-pro@46027a3bffc5817ce8f9b29add0a062610134a90
LOADER_SHA256 = "4a1873d8e84207c8ae19b57e0b4f85916309d33a18ad922ef204de6685b57e0e"

os.environ.update({"MAX_BATCH_SIZE": "4", "BATCH_WINDOW_MS": "0", "BATCH_BUCKET_SECONDS": "2,4,8"})


def _unregister_metrics():
    from prometheus_client import REGISTRY

    for collector, names in list(REGISTRY._collector_to_names.items()):
        if any(name.startswith("parakeet_") for name in names):
            REGISTRY.unregister(collector)


def _load_module():
    _unregister_metrics()
    spec = importlib.util.spec_from_file_location("fluid_speech_model_test", MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Other test modules load model.py at import time without unregistering first.
    _unregister_metrics()
    return module


mm = _load_module()


class AcceleratorError(RuntimeError):
    """Stands in for torch.AcceleratorError (matched by class name)."""


def test_vendored_loader_is_byte_identical_to_the_pin():
    assert hashlib.sha256(LOADER_PATH.read_bytes()).hexdigest() == LOADER_SHA256


def test_is_cuda_error():
    assert mm._is_cuda_error(AcceleratorError("CUDA error: an illegal memory access was encountered"))
    assert mm._is_cuda_error(RuntimeError("CUDA error: device-side assert triggered"))
    # Model Scout E1 (#1377).
    assert mm._is_cuda_error(AcceleratorError("CUDA error: invalid argument"))
    # Found anywhere in the cause chain.
    wrapped = RuntimeError("Parakeet decode pipeline failed")
    wrapped.__cause__ = AcceleratorError("CUDA error: invalid argument")
    assert mm._is_cuda_error(wrapped)
    outer = ValueError("batch failed")
    outer.__context__ = RuntimeError("CUDA error: unspecified launch failure")
    assert mm._is_cuda_error(outer)
    # Not CUDA runtime errors.
    assert not mm._is_cuda_error(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))
    assert not mm._is_cuda_error(ValueError("device-side assert"))
    assert not mm._is_cuda_error(RuntimeError("cuda error"))
    assert not mm._is_cuda_error(None)


def _job(seconds=1.0):
    return {"duration_seconds": seconds, "done": threading.Event()}


def test_other_cuda_error_fails_only_its_batch_when_probe_passes(caplog):
    model = mm.Model()
    model._exit_process = lambda: pytest.fail("must not exit after a recovered CUDA error")
    probes = []
    model._cuda_health_probe = lambda: probes.append(1)
    jobs = [_job(28.68)]
    error = AcceleratorError("CUDA error: invalid argument")
    with caplog.at_level("WARNING"):
        model._finish_jobs(jobs, 0.0, error)
    assert jobs[0]["done"].is_set() and jobs[0]["error"] is error
    assert probes == [1]
    assert model.is_healthy() and model.recovered_cuda_errors == 1
    assert "replica stays up" in caplog.text
    # The next batch is served normally.
    ok = [_job()]
    model._finish_jobs(ok, 0.0, [{"transcript": "hi"}])
    assert ok[0]["output"] == {"transcript": "hi"} and "error" not in ok[0]


def test_other_cuda_error_escalates_when_probe_fails(monkeypatch):
    monkeypatch.setattr(mm, "FATAL_EXIT_DELAY_SECONDS", 0.0)
    model = mm.Model()
    exited = threading.Event()
    model._exit_process = exited.set
    probe_error = AcceleratorError("CUDA error: invalid argument")
    model._cuda_health_probe = lambda: probe_error
    model._finish_jobs([_job()], 0.0, AcceleratorError("CUDA error: invalid argument"))
    assert exited.wait(2)
    assert not model.is_healthy() and model._fatal_error is probe_error


def test_sticky_cuda_error_exits_via_the_probe(monkeypatch):
    # A sticky error poisons the context, so the probe raises the same error again.
    monkeypatch.setattr(mm, "FATAL_EXIT_DELAY_SECONDS", 0.0)
    model = mm.Model()
    exited = threading.Event()
    model._exit_process = exited.set
    error = AcceleratorError("CUDA error: unspecified launch failure")
    model._cuda_health_probe = lambda: error
    model._finish_jobs([_job()], 0.0, error)
    assert exited.wait(2) and not model.is_healthy()


def test_non_cuda_error_does_not_probe():
    model = mm.Model()
    model._cuda_health_probe = lambda: pytest.fail("a non-CUDA error must not be probed")
    jobs = [_job()]
    error = RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
    model._finish_jobs(jobs, 0.0, error)
    assert jobs[0]["error"] is error and model.is_healthy()


def test_failed_batch_log_lists_recent_batches(caplog, monkeypatch):
    monkeypatch.setattr(mm, "RECENT_BATCH_LOG_SIZE", 4)
    model = mm.Model()
    model._cuda_health_probe = lambda: None
    model._exit_process = lambda: pytest.fail("must not exit")
    for index in range(6):
        jobs = [_job(1.0 + index), _job(2.0)]
        waveforms = [types.SimpleNamespace(shape=(16000 * (index + 1),)) for _ in jobs]
        seq = model._record_batch(jobs, waveforms, overlapped=index % 2 == 0)
        for job in jobs:
            job["batch_seq"] = seq
    recent = list(model._recent_batches)
    assert [batch["seq"] for batch in recent] == [3, 4, 5, 6]
    assert recent[-1] == {
        "seq": 6,
        "rows": 2,
        "audio_s": [6.0, 2.0],
        "path": "serial",
        "graph_key": None,
        "timestamps": False,
    }
    assert recent[-2]["path"] == "graph+overlap"
    with caplog.at_level("ERROR"):
        model._finish_jobs(jobs, 0.0, AcceleratorError("CUDA error: invalid argument"))
    assert "(CUDA error)" in caplog.text and "seq=6" in caplog.text
    assert "'seq': 3" in caplog.text and "'seq': 2" not in caplog.text


def test_staging_does_not_resync_an_event_that_raised():
    staging = object.__new__(mm.PinnedStaging)

    class Event:
        calls = 0

        def synchronize(self):
            Event.calls += 1
            raise AcceleratorError("CUDA error: invalid argument")

    staging._signals = [None, None]
    staging._lengths = [None, None]
    staging._copied = [Event(), None]
    staging._next = 0
    with pytest.raises(AcceleratorError):
        staging.stage([], 1, 1)
    assert staging._copied[0] is None and Event.calls == 1


def test_fatal_cuda_error_turns_unhealthy_and_exits_once(monkeypatch):
    monkeypatch.setattr(mm, "FATAL_EXIT_DELAY_SECONDS", 0.0)
    model = mm.Model()
    exited = threading.Event()
    calls = []
    model._exit_process = lambda: (calls.append(1), exited.set())
    assert model.is_healthy()
    jobs = [_job(), _job(2.0)]
    error = AcceleratorError("CUDA error: device-side assert triggered")
    model._cuda_health_probe = lambda: error
    model._finish_jobs(jobs, 0.0, error)
    assert all(job["done"].is_set() and job["error"] is error for job in jobs)
    assert exited.wait(2)
    assert not model.is_healthy()
    # A second failing batch neither re-arms the exit nor clears the state.
    model._finish_jobs([_job()], 0.0, RuntimeError("CUDA error: an illegal memory access"))
    assert calls == [1]
    with pytest.raises(RuntimeError, match="fatal CUDA error"):
        model.predict({"audio_url": "https://example.invalid/a.wav"})


def test_request_errors_do_not_turn_unhealthy():
    model = mm.Model()
    model._exit_process = lambda: pytest.fail("must not exit on a non-CUDA error")
    jobs = [_job()]
    model._finish_jobs(jobs, 0.0, ValueError("could not decode audio"))
    assert jobs[0]["error"] is not None and model.is_healthy()


def test_dead_worker_is_unhealthy():
    model = mm.Model()
    model._worker = types.SimpleNamespace(is_alive=lambda: False)
    assert not model.is_healthy()


def test_fluid_block_check():
    torch = pytest.importorskip("torch")
    nn = torch.nn

    spec = importlib.util.spec_from_file_location("fluid_speech_vendored", LOADER_PATH)
    loader = importlib.util.module_from_spec(spec)
    import sys

    sys.modules.setdefault("nemo", types.ModuleType("nemo"))
    sys.modules.setdefault("nemo.collections", types.ModuleType("nemo.collections"))
    sys.modules.setdefault("nemo.collections.asr", types.ModuleType("nemo.collections.asr"))
    models_stub = types.ModuleType("nemo.collections.asr.models")
    models_stub.ASRModel = object
    sys.modules.setdefault("nemo.collections.asr.models", models_stub)
    spec.loader.exec_module(loader)

    class PreEncode(nn.Module):
        def forward(self, x, lengths):
            return x, lengths

    class Encoder(nn.Module):
        d_model = 8

        def __init__(self):
            super().__init__()
            self.pre_encode = PreEncode()
            self.layers = nn.ModuleList([nn.Linear(8, 8)])

        def forward(self, x, lengths):
            x, lengths = self.pre_encode(x=x, lengths=lengths)
            return self.layers[0](x), lengths

    model = types.SimpleNamespace(encoder=Encoder())
    with pytest.raises(RuntimeError, match="no encoder.fv_insert"):
        mm.Model._check_fluid_block(model, "load")
    block = loader.attach(model, 4, 3)
    mm.Model._check_fluid_block(model, "load")
    seen = []
    block.register_forward_hook(lambda *_: seen.append(1))
    x = torch.randn(2, 5, 8)
    out, _ = model.encoder(x, torch.tensor([5, 3]))
    assert seen == [1] and out.shape == x.shape
    # A cast of the encoder moves the block too; a block left behind fails the check.
    model.encoder.to(torch.bfloat16)
    mm.Model._check_fluid_block(model, "cast")
    block.to(torch.float32)
    with pytest.raises(RuntimeError, match="dtype"):
        mm.Model._check_fluid_block(model, "cast")
