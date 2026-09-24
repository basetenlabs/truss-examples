"""Long-audio routing, input caps, batch-failure isolation and health in model/model.py.

Not run by this repository's CI, which has no Python test job -- run by hand with

    uv run --with pytest --with numpy --with soundfile --with prometheus-client \
        --with fastapi pytest stt/orukeet/latency/tests

NeMo and a GPU are stubbed throughout: the checkpoint cannot be downloaded outside
Baseten and a CUDA context cannot be created here. What these tests cover is the
runtime's own logic -- which model and batch size each duration gets, what is rejected,
how a failed batch is split, what a failure leaves pinned, and when the replica reports
unhealthy. The real attention switch, memory use and accuracy are exercised by the
deployment's startup capacity check and the batch benchmark.
"""

from __future__ import annotations

import base64
import gc
import importlib.util
import io
import os
import shutil
import subprocess
import sys
import threading
import types
import weakref
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "model.py"
SR = 16_000

# The deployed config's values for everything the module reads at import time.
os.environ.update(
    {
        "MAX_BATCH_SIZE": "16",
        "BATCH_WINDOW_MS": "5",
        "BATCH_BUCKET_SECONDS": "2,4,8",
        "PARAKEET_DIRECT_FORWARD": "true",
        "FULL_ATTENTION_MAX_SECONDS": "1440",
        "MAX_AUDIO_SECONDS": "10800",
    }
)
_spec = importlib.util.spec_from_file_location("orukeet_model", MODEL_PATH)
mm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mm)


def _status(exc: BaseException):
    return getattr(exc, "status_code", None)


class FakeModel:
    def __init__(self, name: str):
        self.name = name


class Recorder:
    """Stands in for Model._infer_waveforms; records every batch it is given."""

    def __init__(self, fail=None):
        self.batches = []
        self.lock = threading.Lock()
        self.fail = fail  # callable(model, waveforms) -> Exception | None

    def __call__(self, model, waveforms, *, with_timestamps):
        lengths = [w.shape[0] for w in waveforms]
        with self.lock:
            self.batches.append((model.name, lengths))
        if self.fail is not None:
            error = self.fail(model, waveforms)
            if error is not None:
                raise error
        return [
            types.SimpleNamespace(text=f"{model.name}:{n}", timestamp={}) for n in lengths
        ]


@pytest.fixture
def runtime(monkeypatch):
    model = mm.Model()
    model._model = FakeModel("full")
    model._long_model = FakeModel("local")
    recorder = Recorder()
    model._infer_waveforms = recorder
    # A request carries a sample count; decode returns a zero-cost view of that length.
    model._decode_audio_input = lambda request: request["samples"]
    monkeypatch.setattr(
        mm.Model,
        "_decode_waveform",
        staticmethod(lambda n: np.broadcast_to(np.float32(0), (n,))),
    )
    monkeypatch.setattr(
        mm.Model, "_cuda_context_usable", staticmethod(lambda **_: True)
    )
    model._worker = threading.Thread(target=model._batch_worker, daemon=True)
    model._worker.start()
    return model, recorder


def seconds(value: float) -> int:
    return round(value * SR)


# --- routing -----------------------------------------------------------------------


def test_duration_routes_to_the_right_attention_mode(runtime):
    model, recorder = runtime
    cases = {
        30: "full",
        300: "full",
        1440: "full",  # exactly the full-attention limit stays on full attention
        1441: "local",
        3600: "local",
        10800: "local",  # exactly the cap is accepted
    }
    for duration, expected in cases.items():
        result = model.predict({"samples": seconds(duration)})
        assert result["text"] == f"{expected}:{seconds(duration)}", duration


def test_audio_over_the_cap_is_rejected_before_any_gpu_work(runtime):
    model, recorder = runtime
    with pytest.raises(Exception) as caught:
        model.predict({"samples": seconds(10800) + 1})
    assert _status(caught.value) == 413
    assert recorder.batches == []
    assert model.is_healthy()


def test_short_clip_buckets_and_batch_caps_are_unchanged():
    # The T4-tuned short path: <=8 s clips keep their buckets and MAX_BATCH_SIZE.
    assert [mm.Model._duration_bucket(d) for d in (0.5, 2, 3, 8, 9, 30, 60)] == [
        2.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0,
    ]
    for bucket in (2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0):
        assert mm.Model._max_batch_size(bucket) == 16


def test_long_full_attention_batches_are_capped_by_quadratic_memory():
    # B * bucket^2 <= 1440^2: 512 s -> 7, 1024 s and 2048 s -> 1, local attention -> 1.
    assert mm.Model._duration_bucket(300) == 512.0
    assert mm.Model._duration_bucket(1440) == 2048.0
    assert mm.Model._duration_bucket(1441) == mm.LONG_AUDIO_BUCKET
    assert mm.Model._max_batch_size(512.0) == 7
    assert mm.Model._max_batch_size(1024.0) == 1
    assert mm.Model._max_batch_size(2048.0) == 1
    assert mm.Model._max_batch_size(mm.LONG_AUDIO_BUCKET) == 1
    for bucket in (512.0, 1024.0, 2048.0):
        assert mm.Model._max_batch_size(bucket) * bucket**2 <= 1440**2 or (
            mm.Model._max_batch_size(bucket) == 1
        )


def test_concurrent_mixed_lengths_never_mix_models_or_exceed_caps(runtime):
    model, recorder = runtime
    durations = [3] * 24 + [30] * 8 + [300] * 12 + [1200] * 3 + [3600] * 4 + [10800] * 2
    results, errors = {}, []

    def call(index, duration):
        try:
            results[index] = model.predict({"samples": seconds(duration)})
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [
        threading.Thread(target=call, args=(i, d)) for i, d in enumerate(durations)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not errors
    for index, duration in enumerate(durations):
        mode = "local" if duration > 1440 else "full"
        assert results[index]["text"] == f"{mode}:{seconds(duration)}"

    for name, lengths in recorder.batches:
        longest = max(lengths) / SR
        if name == "local":
            assert len(lengths) == 1 and longest > 1440
        else:
            assert longest <= 1440
            bucket = mm.Model._duration_bucket(longest)
            assert len(lengths) <= mm.Model._max_batch_size(bucket)
            # Every clip in a batch shares one bucket.
            assert {mm.Model._duration_bucket(n / SR) for n in lengths} == {bucket}


# --- failure isolation and health --------------------------------------------------

POISON = seconds(3.3)


def test_one_failing_job_does_not_fail_its_batch(runtime):
    model, recorder = runtime
    recorder.fail = lambda m, ws: (
        ValueError("bad input") if any(w.shape[0] == POISON for w in ws) else None
    )
    samples = [seconds(3)] * 6 + [POISON]
    outcomes = [None] * len(samples)
    gate = threading.Barrier(len(samples))

    def call(index):
        gate.wait()
        try:
            outcomes[index] = model.predict({"samples": samples[index]})["text"]
        except Exception as exc:
            outcomes[index] = exc

    threads = [threading.Thread(target=call, args=(i,)) for i in range(len(samples))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert outcomes[:6] == [f"full:{seconds(3)}"] * 6
    assert isinstance(outcomes[6], mm.InferenceError)
    assert "bad input" in str(outcomes[6])
    assert model.is_healthy()
    # The shared batch failed once, then each job ran alone.
    assert any(len(lengths) > 1 for _, lengths in recorder.batches)


class Activation:
    """Stands in for a GPU tensor held by a frame of a failed forward pass."""


def test_a_failed_pass_does_not_pin_its_activations(runtime):
    model, recorder = runtime
    pinned = []

    def forward_that_ooms():
        activation = Activation()
        pinned.append(weakref.ref(activation))
        raise RuntimeError("CUDA out of memory. Tried to allocate 60.33 GiB")

    recorder.fail = lambda m, ws: forward_that_ooms()
    gc.disable()  # reference counting alone must release it, as with frozen GC
    try:
        with pytest.raises(mm.InferenceError) as caught:
            model.predict({"samples": seconds(3)})
        assert "out of memory" in str(caught.value)
        del caught
        assert pinned and pinned[0]() is None, "failed pass still pinned by a traceback"
    finally:
        gc.enable()
    assert model.is_healthy()
    # The replica keeps serving after a request-scoped OOM.
    recorder.fail = None
    assert model.predict({"samples": seconds(3)})["text"] == f"full:{seconds(3)}"


def test_poisoned_cuda_context_marks_replica_unhealthy(runtime):
    model, recorder = runtime
    recorder.fail = lambda m, ws: RuntimeError(
        "CUDA error: an illegal memory access was encountered"
    )
    with pytest.raises(mm.InferenceError):
        model.predict({"samples": seconds(3)})
    assert not model.is_healthy()

    recorder.fail = None
    before = len(recorder.batches)
    with pytest.raises(Exception) as caught:
        model.predict({"samples": seconds(3)})
    assert _status(caught.value) == 503
    assert len(recorder.batches) == before, "a poisoned replica must not run inference"


def test_failed_probe_marks_replica_unhealthy(runtime, monkeypatch):
    model, recorder = runtime
    monkeypatch.setattr(
        mm.Model, "_cuda_context_usable", staticmethod(lambda **_: False)
    )
    recorder.fail = lambda m, ws: RuntimeError("CUBLAS error of some new kind")
    with pytest.raises(mm.InferenceError):
        model.predict({"samples": seconds(3)})
    assert not model.is_healthy()


def test_dead_worker_marks_replica_unhealthy():
    model = mm.Model()
    model._worker = threading.Thread(target=lambda: None)
    model._worker.start()
    model._worker.join()
    assert not model.is_healthy()


# --- decoding caps -----------------------------------------------------------------


def wav_bytes(duration_s: float, rate: int) -> bytes:
    buffer = io.BytesIO()
    t = np.arange(round(duration_s * rate)) / rate
    sf.write(buffer, (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), rate, format="WAV")
    return buffer.getvalue()


def test_wav_over_the_cap_is_rejected_from_its_header(monkeypatch):
    monkeypatch.setattr(mm, "MAX_AUDIO_SECONDS", 3.0)
    with pytest.raises(Exception) as caught:
        mm.Model._decode_waveform(wav_bytes(5, SR))
    assert _status(caught.value) == 413
    assert mm.Model._decode_waveform(wav_bytes(2, SR)).shape[0] == 2 * SR


needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def m4a_bytes(tmp_path, duration_s: float) -> bytes:
    """AAC in MP4 with ffmpeg's default moov-at-end layout, like the podcast clips."""
    source = tmp_path / "in.wav"
    source.write_bytes(wav_bytes(duration_s, 44_100))
    target = tmp_path / "out.m4a"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(source), "-c:a", "aac", str(target)],
        check=True,
    )
    return target.read_bytes()


@needs_ffmpeg
def test_ffmpeg_stops_decoding_one_second_past_the_cap(monkeypatch, tmp_path):
    monkeypatch.setattr(mm, "MAX_AUDIO_SECONDS", 3.0)
    # soundfile cannot read AAC, so this goes through ffmpeg, which has no header
    # duration to check first: -t stops it at the cap plus one second.
    waveform = mm.Model._decode_waveform(m4a_bytes(tmp_path, 10))
    assert abs(waveform.shape[0] - 4 * SR) <= SR // 20
    assert waveform.shape[0] / SR > mm.MAX_AUDIO_SECONDS  # predict() then rejects it


@needs_ffmpeg
def test_ffmpeg_decodes_moov_at_end_m4a(tmp_path):
    waveform = mm.Model._decode_waveform(m4a_bytes(tmp_path, 20))
    assert waveform.dtype == np.float32
    assert abs(waveform.shape[0] / SR - 20) < 0.1


def test_resampled_wav_over_the_cap_is_rejected_from_its_header(monkeypatch):
    monkeypatch.setattr(mm, "MAX_AUDIO_SECONDS", 3.0)
    with pytest.raises(Exception) as caught:
        mm.Model._decode_waveform(wav_bytes(5, 8_000))
    assert _status(caught.value) == 413


def test_oversized_b64_input_is_rejected(monkeypatch):
    monkeypatch.setattr(mm, "MAX_AUDIO_INPUT_BYTES", 1000)
    model = mm.Model()
    small = base64.b64encode(b"x" * 999).decode()
    large = base64.b64encode(b"x" * 1001).decode()
    assert len(model._decode_audio_input({"audio_b64": small})) == 999
    with pytest.raises(Exception) as caught:
        model._decode_audio_input({"audio_b64": large})
    assert _status(caught.value) == 413


# --- short-only deployments (MAX_AUDIO_SECONDS == FULL_ATTENTION_MAX_SECONDS) -------


def _import_model(monkeypatch, name: str, **env):
    """Import a fresh copy of model.py under `env`, since the limits are read at import.

    prometheus_client is stubbed for the copy: its metrics are process-global, and the
    module above already registered them.
    """
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    class _Metric:
        def __init__(self, *args, **kwargs):
            pass

        def labels(self, **kwargs):
            return self

        def observe(self, value):
            pass

        def inc(self):
            pass

        def dec(self):
            pass

    stub = types.ModuleType("prometheus_client")
    stub.Gauge = stub.Histogram = _Metric
    monkeypatch.setitem(sys.modules, "prometheus_client", stub)
    spec = importlib.util.spec_from_file_location(name, MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def short_only(monkeypatch):
    module = _import_model(
        monkeypatch,
        "orukeet_model_short_only",
        FULL_ATTENTION_MAX_SECONDS="1440",
        MAX_AUDIO_SECONDS="1440",
    )
    assert module.LONG_AUDIO_ENABLED is False
    return module


class _Context:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Tensor:
    def __init__(self, shape):
        self.shape = shape

    def to(self, **kwargs):
        return self

    def __getitem__(self, key):
        assert key is None  # waveform[None]: add the batch dimension
        return _Tensor((1, *self.shape))


def _fake_load_stack(monkeypatch, module):
    """Stub torch / nemo / httpx / omegaconf so load() runs on the CPU.

    The capacity check allocates a zero waveform of its full length (3 hr at 16 kHz is
    ~690 MB), so the sample rate drops to 10 Hz: the check still runs, on tiny arrays.
    """
    cuda = types.SimpleNamespace(
        get_device_name=lambda *a: "Fake GPU",
        get_device_capability=lambda *a: (12, 0),
        get_device_properties=lambda *a: types.SimpleNamespace(total_memory=96 * 2**30),
        synchronize=lambda: None,
        reset_peak_memory_stats=lambda: None,
        max_memory_allocated=lambda: 0,
        max_memory_reserved=lambda: 0,
        empty_cache=lambda: None,
    )
    torch = types.ModuleType("torch")
    torch.__version__ = "0+fake"
    torch.version = types.SimpleNamespace(cuda="12.8")
    torch.backends = types.SimpleNamespace(
        cudnn=types.SimpleNamespace(enabled=True, version=lambda: 91000)
    )
    torch.cuda = cuda
    torch.float16 = "float16"
    torch.inference_mode = torch.autocast = _Context
    torch.from_numpy = lambda array: _Tensor(array.shape)
    torch.tensor = lambda values, **kwargs: _Tensor((len(values),))
    omegaconf = types.ModuleType("omegaconf")
    omegaconf.open_dict = _Context
    httpx = types.ModuleType("httpx")
    httpx.Client = lambda **kwargs: object()
    for name, stub in (
        ("torch", torch),
        ("nemo", types.ModuleType("nemo")),
        ("omegaconf", omegaconf),
        ("httpx", httpx),
    ):
        monkeypatch.setitem(sys.modules, name, stub)
    monkeypatch.setattr(module, "AUDIO_SAMPLE_RATE_HZ", 10)
    monkeypatch.setitem(module._TIMING_STATE, "serving", False)


class LoadableModel:
    """A restored checkpoint: records attention switches and encoder-only passes."""

    def __init__(self, index: int):
        self.name = "full" if index == 0 else "local"
        self.encoder = types.SimpleNamespace(
            self_attention_model="rel_pos", att_context_size=[-1, -1]
        )
        self.cfg = types.SimpleNamespace(
            decoding=types.SimpleNamespace(
                greedy=types.SimpleNamespace(use_cuda_graph_decoder=True)
            )
        )
        self.encoder_passes = []

    def change_attention_model(self, self_attention_model, att_context_size):
        self.encoder.self_attention_model = self_attention_model
        self.encoder.att_context_size = att_context_size

    def change_subsampling_conv_chunking_factor(self, factor):
        pass

    def change_decoding_strategy(self, cfg, verbose=True):
        pass

    def forward(self, input_signal, input_signal_length):
        self.encoder_passes.append(input_signal.shape)
        return object(), object()


def _run_load(monkeypatch, module):
    _fake_load_stack(monkeypatch, module)
    restored = []

    def restore():
        model = LoadableModel(len(restored))
        restored.append(model)
        return model

    monkeypatch.setattr(module.Model, "_restore_model", staticmethod(restore))
    runtime = module.Model()
    recorder = Recorder()
    runtime._infer_waveforms = recorder
    runtime.load()
    return runtime, restored, recorder


def test_short_only_load_restores_one_model_and_skips_long_form(
    monkeypatch, short_only, caplog
):
    caplog.set_level("INFO")
    runtime, restored, recorder = _run_load(monkeypatch, short_only)
    assert len(restored) == 1
    assert runtime._model is restored[0]
    assert runtime._long_model is None
    # The full-attention capacity check still ran, encoder only, at 24 min.
    assert restored[0].encoder_passes == [(1, 1440 * 10)]
    # No local-attention transcription ran at load.
    assert recorder.batches == []
    assert "Parakeet long-form model disabled" in caplog.text
    assert "mode=local_attention" not in caplog.text
    assert runtime._cudnn_for(runtime._model) == short_only.SHORT_PATH_CUDNN
    assert runtime.is_healthy()


def test_default_load_still_restores_both_models(monkeypatch, caplog):
    caplog.set_level("INFO")
    assert mm.LONG_AUDIO_ENABLED is True
    runtime, restored, recorder = _run_load(monkeypatch, mm)
    assert len(restored) == 2
    full, local = restored
    assert runtime._model is full and runtime._long_model is local
    assert full.encoder.self_attention_model == "rel_pos"
    assert local.encoder.self_attention_model == "rel_pos_local_attn"
    assert local.encoder.att_context_size == [256, 256]
    assert local.cfg.decoding.greedy.use_cuda_graph_decoder is False
    # Both capacity checks ran: full attention encoder-only, local attention decoded.
    assert full.encoder_passes == [(1, 1440 * 10)]
    assert recorder.batches == [("local", [10800 * 10])]
    assert "mode=full_attention" in caplog.text
    assert "mode=local_attention" in caplog.text
    assert "Parakeet long-form model disabled" not in caplog.text
    assert runtime._cudnn_for(local) == mm.LONG_PATH_CUDNN
    assert runtime.is_healthy()


def test_short_only_rejects_30_min_and_serves_10_min(monkeypatch, short_only):
    model = short_only.Model()
    model._model = FakeModel("full")
    recorder = Recorder()
    model._infer_waveforms = recorder
    model._decode_audio_input = lambda request: request["samples"]
    monkeypatch.setattr(
        short_only.Model,
        "_decode_waveform",
        staticmethod(lambda n: np.broadcast_to(np.float32(0), (n,))),
    )
    monkeypatch.setattr(
        short_only.Model, "_cuda_context_usable", staticmethod(lambda **_: True)
    )
    model._worker = threading.Thread(target=model._batch_worker, daemon=True)
    model._worker.start()
    assert model._long_model is None

    with pytest.raises(Exception) as caught:
        model.predict({"samples": seconds(30 * 60)})
    assert _status(caught.value) == 413
    assert recorder.batches == []

    for duration in (10 * 60, 1440):
        result = model.predict({"samples": seconds(duration)})
        assert result["text"] == f"full:{seconds(duration)}"
    assert [name for name, _ in recorder.batches] == ["full", "full"]
    assert model.is_healthy()


def test_short_only_refuses_a_long_route_instead_of_using_none(short_only):
    # predict()'s 413 keeps this unreachable; a misroute must fail loudly.
    with pytest.raises(RuntimeError, match="not loaded"):
        short_only.Model._duration_bucket(1441)
    model = short_only.Model()
    model._model = FakeModel("full")
    with pytest.raises(RuntimeError, match="no long-form model"):
        model._run_batch(
            [{"bucket": short_only.LONG_AUDIO_BUCKET, "waveform": None, "timestamps": False}],
            isolate_failures=False,
        )
