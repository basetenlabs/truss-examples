"""NeMo's transcribe() leaves the model in training mode; model/model.py restores it.

Not run by this repository's CI, which has no Python test job -- run by hand with

    uv run --with pytest --with numpy --with prometheus-client pytest stt/parakeet-tdt-0.6b-v3/batch/tests/test_transcribe_mode.py

NeMo and torch are stubbed: FakeModel.transcribe() does what NeMo 2.7.3 / 3.0.0's ASR
transcribe() does on return (encoder/decoder/joint.unfreeze() -> module.train(), and the
checkpoint's dither / pad_to restored), and the tests check that the timestamps path
puts the model back in eval mode with the direct path's featurizer settings.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "model.py"


def _load_module():
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
    from prometheus_client import REGISTRY

    for collector, names in list(REGISTRY._collector_to_names.items()):
        if any(name.startswith("parakeet_") for name in names):
            REGISTRY.unregister(collector)
    spec = importlib.util.spec_from_file_location("transcribe_mode_model", MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def mm(monkeypatch):
    torch = types.ModuleType("torch")
    torch.float16 = "float16"
    torch.inference_mode = contextlib.nullcontext
    torch.autocast = lambda **_kwargs: contextlib.nullcontext()
    torch.backends = types.SimpleNamespace(cudnn=types.SimpleNamespace(enabled=False))
    monkeypatch.setitem(sys.modules, "torch", torch)
    return _load_module()


class Module:
    def __init__(self):
        self.training = False


class FakeModel:
    def __init__(self):
        self.encoder, self.decoder, self.joint = Module(), Module(), Module()
        self.training = False
        self.preprocessor = types.SimpleNamespace(
            featurizer=types.SimpleNamespace(dither=0.0, pad_to=0)
        )
        self.transcribe_calls = 0

    def eval(self):
        self.training = False
        for module in (self.encoder, self.decoder, self.joint):
            module.training = False
        return self

    def transcribe(self, waveforms, **_kwargs):
        self.transcribe_calls += 1
        # What NeMo's _transcribe_on_end() leaves behind.
        for module in (self.encoder, self.decoder, self.joint):
            module.training = True
        self.preprocessor.featurizer.dither = 1e-5
        self.preprocessor.featurizer.pad_to = 16
        return [f"text {i}" for i in range(len(waveforms))]


def test_restore_eval_after_transcribe(mm):
    model = FakeModel()
    model.transcribe(["a"])
    assert mm._restore_eval_after_transcribe(model) == ["encoder", "decoder", "joint"]
    assert not any(m.training for m in (model.encoder, model.decoder, model.joint))
    assert model.preprocessor.featurizer.dither == 0.0
    assert model.preprocessor.featurizer.pad_to == 0


def test_timestamps_path_leaves_model_in_eval_mode(mm):
    model = FakeModel()
    instance = mm.Model()
    instance._model = model
    out = instance._infer_waveforms(["w"], with_timestamps=True)
    assert out == ["text 0"]
    assert model.transcribe_calls == 1
    assert not any(m.training for m in (model.encoder, model.decoder, model.joint))
    assert model.preprocessor.featurizer.pad_to == 0


def test_startup_mode_check_fails_if_training_mode_remains(mm, monkeypatch):
    model = FakeModel()
    instance = mm.Model()
    instance._model = model
    instance._check_transcribe_mode(model)  # restored: passes
    monkeypatch.setattr(mm, "_restore_eval_after_transcribe", lambda _model: [])
    with pytest.raises(RuntimeError, match="training mode"):
        instance._check_transcribe_mode(model)


class FakeDecoding:
    """The TDT decoding object; the graph mode is what _log_decoder_graph_mode reads."""

    def __init__(self, compute_timestamps, mode="full_graph"):
        self.compute_timestamps = compute_timestamps
        computer = types.SimpleNamespace(cuda_graphs_mode=mode)
        self.decoding = types.SimpleNamespace(decoding_computer=computer)


class Cfg(dict):
    pass


class TimestampsModel(FakeModel):
    """FakeModel plus what NeMo 3.0.0's EncDecRNNTModel.transcribe(timestamps=...)
    does to the decoding setup (rnnt_models.py): when cfg.decoding.compute_timestamps
    is not the requested value it sets it and calls change_decoding_strategy(), which
    builds a new decoding object."""

    def __init__(self):
        super().__init__()
        self.cfg = types.SimpleNamespace(decoding=Cfg(compute_timestamps=None))
        self.decoding = FakeDecoding(None)
        self.rebuilds = []
        self.next_mode = "full_graph"

    def change_decoding_strategy(self, decoding_cfg, verbose=True):
        self.cfg.decoding = Cfg(decoding_cfg)
        self.decoding = FakeDecoding(decoding_cfg.get("compute_timestamps"), self.next_mode)
        self.rebuilds.append(decoding_cfg.get("compute_timestamps"))

    def transcribe(self, waveforms, timestamps=None, **kwargs):
        if timestamps is not None and self.cfg.decoding.get("compute_timestamps") is not timestamps:
            self.change_decoding_strategy(
                Cfg(self.cfg.decoding, compute_timestamps=bool(timestamps))
            )
        return super().transcribe(waveforms, **kwargs)


def _serving_instance(mm, monkeypatch):
    """A Model as load() leaves it: decoding snapshotted, warmed, enforced."""
    monkeypatch.setattr(mm, "REQUIRE_FULL_CUDA_GRAPH", True)
    model = TimestampsModel()
    instance = mm.Model()
    instance._model = model
    instance._decoding_cfg = Cfg(model.cfg.decoding)
    instance._serving_decoding = model.decoding
    events = []
    monkeypatch.setattr(
        instance,
        "_warmup",
        lambda *, decoder_only=False: events.append(("warmup", decoder_only)),
    )

    def infer_direct(waveforms):
        decoding = model.decoding
        events.append(("direct", decoding.compute_timestamps, decoding is instance._serving_decoding))
        return [f"direct {i}" for i in range(len(waveforms))]

    monkeypatch.setattr(instance, "_infer_direct", infer_direct)
    return model, instance, events


def test_direct_batch_after_timestamps_runs_on_rebuilt_warmed_decoding(mm, monkeypatch):
    model, instance, events = _serving_instance(mm, monkeypatch)
    warmed = model.decoding

    instance._infer_waveforms(["w"], with_timestamps=True)
    # NeMo replaced the decoding object with a compute_timestamps=True one.
    assert model.decoding is not warmed and model.decoding.compute_timestamps is True
    timestamps_decoding = model.decoding

    assert instance._infer_waveforms(["a", "b"], with_timestamps=False) == ["direct 0", "direct 1"]
    # Rebuilt from the load-time config (not the old object: unsafe on TDT), warmed
    # (decoder graph only) before the batch, and the batch decoded on it.
    assert model.rebuilds == [True, None]
    assert model.cfg.decoding == {"compute_timestamps": None}
    assert model.decoding is not timestamps_decoding and model.decoding is not warmed
    assert instance._serving_decoding is model.decoding
    assert events == [("warmup", True), ("direct", None, True)]


def test_timestamps_batches_back_to_back_rebuild_nothing_extra(mm, monkeypatch):
    model, instance, events = _serving_instance(mm, monkeypatch)
    for _ in range(3):
        instance._infer_waveforms(["w"], with_timestamps=True)
    # Only NeMo's own first switch to timestamps; no restore between them.
    assert model.rebuilds == [True]
    assert events == []
    instance._infer_waveforms(["a"], with_timestamps=False)
    instance._infer_waveforms(["a"], with_timestamps=False)
    # One restore for the run, then direct batches reuse the restored object.
    assert model.rebuilds == [True, None]
    assert events == [("warmup", True), ("direct", None, True), ("direct", None, True)]


def test_restore_enforces_full_graph_and_retries_until_verified(mm, monkeypatch):
    model, instance, events = _serving_instance(mm, monkeypatch)
    instance._infer_waveforms(["w"], with_timestamps=True)
    model.next_mode = "no_while_loops"
    with pytest.raises(RuntimeError, match="REQUIRE_FULL_CUDA_GRAPH"):
        instance._infer_waveforms(["a"], with_timestamps=False)
    assert instance._serving_decoding is None
    # The next direct batch does not serve on the unverified object: it rebuilds again.
    model.next_mode = "full_graph"
    assert instance._infer_waveforms(["a"], with_timestamps=False) == ["direct 0"]
    assert model.rebuilds == [True, None, None]
    assert events[-1] == ("direct", None, True)


def test_direct_batches_without_timestamps_never_rebuild(mm, monkeypatch):
    model, instance, events = _serving_instance(mm, monkeypatch)
    warmed = model.decoding
    for _ in range(3):
        instance._infer_waveforms(["a"], with_timestamps=False)
    assert model.rebuilds == []
    assert model.decoding is warmed
    assert events == [("direct", None, True)] * 3


def test_decoder_only_warmup_runs_only_the_longest_clip(mm, monkeypatch):
    monkeypatch.setattr(mm, "WARMUP_AUDIO_SECONDS", (0.15, 6.0, 1.6))
    monkeypatch.setattr(mm, "WARMUP_BATCH_SIZE", 2)
    instance = mm.Model()
    seen = []
    monkeypatch.setattr(
        instance,
        "_infer_waveforms",
        lambda waveforms, *, with_timestamps: seen.append((len(waveforms), waveforms[0].shape[0])),
    )
    instance._warmup(decoder_only=True)
    assert seen == [(2, 6 * mm.AUDIO_SAMPLE_RATE_HZ)]
    seen.clear()
    instance._warmup()
    assert [samples for _, samples in seen] == [
        round(s * mm.AUDIO_SAMPLE_RATE_HZ) for s in (0.15, 6.0, 1.6)
    ]
