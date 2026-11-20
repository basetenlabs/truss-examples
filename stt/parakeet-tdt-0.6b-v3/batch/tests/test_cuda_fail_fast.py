"""The CUDA health probe and fail-fast after any CUDA error, the
recent-batch log, and the PinnedStaging event reset in model/model.py. Ported with the
code from stt/fluid-speech-pro/batch-bf16 (#1377).

Not run by this repository's CI, which has no Python test job -- run by hand with

    uv run --with pytest --with numpy --with prometheus-client \
        pytest stt/parakeet-tdt-0.6b-v3/batch/tests/test_cuda_fail_fast.py

CPU only; torch is not needed (CUDA errors are stand-ins matched by class name and
message).
"""

from __future__ import annotations

import importlib.util
import os
import threading
import types
from pathlib import Path

import pytest

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "model.py"

os.environ.update({"MAX_BATCH_SIZE": "4", "BATCH_WINDOW_MS": "0", "BATCH_BUCKET_SECONDS": "2,4,8"})


def _unregister_metrics():
    from prometheus_client import REGISTRY

    for collector, names in list(REGISTRY._collector_to_names.items()):
        if any(name.startswith("parakeet_") for name in names):
            REGISTRY.unregister(collector)


def _load_module():
    _unregister_metrics()
    spec = importlib.util.spec_from_file_location("parakeet_cuda_fail_fast_test", MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Other test modules load model.py at import time without unregistering first.
    _unregister_metrics()
    return module


mm = _load_module()


class AcceleratorError(RuntimeError):
    """Stands in for torch.AcceleratorError (matched by class name)."""


def test_is_cuda_error():
    assert mm._is_cuda_error(AcceleratorError("CUDA error: an illegal memory access was encountered"))
    assert mm._is_cuda_error(RuntimeError("CUDA error: device-side assert triggered"))
    # Model Scout E1 (#1377, Fluid Speech Pro).
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
