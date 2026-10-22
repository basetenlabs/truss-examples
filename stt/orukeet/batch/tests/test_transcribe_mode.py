"""NeMo's transcribe() leaves the model in training mode; model/model.py restores it.

Not run by this repository's CI, which has no Python test job -- run by hand with

    uv run --with pytest --with numpy --with prometheus-client pytest stt/orukeet/batch/tests/test_transcribe_mode.py

NeMo and torch are stubbed: FakeModel.transcribe() does what NeMo 2.7.3 / 3.0.0's ASR
transcribe() does on return (encoder/decoder/joint.unfreeze() -> module.train(), and the
checkpoint's dither / pad_to restored, and with timestamps a rebuilt decoding object),
and the tests check that the timestamps path puts the model back in eval mode with the
direct path's featurizer settings, and that the next direct-path batch runs on a rebuilt,
re-warmed and graph-checked load-time decoding object.
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
        self.decoding = object()
        self.cfg = types.SimpleNamespace(decoding={"compute_timestamps": False})

    def eval(self):
        self.training = False
        for module in (self.encoder, self.decoder, self.joint):
            module.training = False
        return self

    def transcribe(self, waveforms, timestamps=None, **_kwargs):
        self.transcribe_calls += 1
        if timestamps:
            # change_decoding_strategy() with compute_timestamps=True.
            self.cfg.decoding = {"compute_timestamps": True}
            self.decoding = object()
        # What NeMo's _transcribe_on_end() leaves behind.
        for module in (self.encoder, self.decoder, self.joint):
            module.training = True
        self.preprocessor.featurizer.dither = 1e-5
        self.preprocessor.featurizer.pad_to = 16
        return [f"text {i}" for i in range(len(waveforms))]

    def change_decoding_strategy(self, decoding_cfg, verbose=True):
        self.cfg.decoding = dict(decoding_cfg)
        self.decoding = object()
        self.rebuilt_with = dict(decoding_cfg)


def test_restore_eval_after_transcribe(mm):
    model = FakeModel()
    model.transcribe(["a"])
    assert mm._restore_eval_after_transcribe(model) == ["encoder", "decoder", "joint"]
    assert not any(m.training for m in (model.encoder, model.decoder, model.joint))
    assert model.preprocessor.featurizer.dither == 0.0
    assert model.preprocessor.featurizer.pad_to == 0


def _instance(mm, monkeypatch, model, graph_mode="full_graph"):
    """A Model serving `model` as its short model, with the direct path and the
    decoder graph-mode probe stubbed and every warmup call recorded."""
    instance = mm.Model()
    instance._model = model
    instance._long_model = None
    instance.warmups = []
    instance.graph_mode = graph_mode
    monkeypatch.setattr(instance, "_cudnn_for", lambda _model: False)
    monkeypatch.setattr(
        instance, "_infer_direct", lambda m, waveforms: [f"direct {len(waveforms)}"]
    )
    real_warmup = instance._warmup

    def warmup(*, decoder_only=False):
        instance.warmups.append(decoder_only)
        real_warmup(decoder_only=decoder_only)

    monkeypatch.setattr(instance, "_warmup", warmup)

    def graph_mode_check(phase, *, enforce, model=None):
        instance.graph_checks.append((phase, enforce))
        if enforce and instance.graph_mode != "full_graph":
            raise RuntimeError("REQUIRE_FULL_CUDA_GRAPH is enabled")

    instance.graph_checks = []
    monkeypatch.setattr(instance, "_log_decoder_graph_mode", graph_mode_check)
    monkeypatch.setattr(mm, "WARMUP_AUDIO_SECONDS", (0.15, 0.4, 6.0))
    monkeypatch.setattr(mm, "REQUIRE_FULL_CUDA_GRAPH", True)
    mm._snapshot_decoding(model)
    return instance


def test_timestamps_path_leaves_model_in_eval_mode(mm, monkeypatch):
    model = FakeModel()
    instance = _instance(mm, monkeypatch, model)
    out = instance._infer_waveforms(model, ["w"], with_timestamps=True)
    assert out == ["text 0"]
    assert model.transcribe_calls == 1
    assert not any(m.training for m in (model.encoder, model.decoder, model.joint))
    assert model.preprocessor.featurizer.pad_to == 0
    # The decoding rebuild waits for the next direct-path batch.
    assert model.cfg.decoding == {"compute_timestamps": True}


def test_direct_batch_after_timestamps_runs_on_rebuilt_warmed_decoding(mm, monkeypatch):
    import numpy as np

    model = FakeModel()
    instance = _instance(mm, monkeypatch, model)
    warmed = model.decoding
    instance._infer_waveforms(model, ["w"], with_timestamps=True)
    seen = []
    monkeypatch.setattr(
        instance,
        "_infer_direct",
        lambda m, waveforms: (
            seen.append((m.cfg.decoding, len(waveforms), waveforms[0].shape[0])) or ["direct"]
        ),
    )
    customer = [np.zeros(16, dtype=np.float32)]
    assert instance._infer_waveforms(model, customer, with_timestamps=False) == ["direct"]
    # Rebuilt from the load-time config (timestamps off), not the timestamps object.
    assert model.rebuilt_with == {"compute_timestamps": False}
    assert model.decoding is not warmed
    assert mm._SERVING_DECODING[id(model)][0] is model.decoding
    # Warmed on the rebuilt decoder (longest clip only) and graph-checked before
    # the customer batch, which is served last.
    assert instance.warmups == [True]
    assert instance.graph_checks == [("rewarm_after_timestamps", True)]
    longest = round(6.0 * mm.AUDIO_SAMPLE_RATE_HZ)
    assert [entry[2] for entry in seen[:-1]] == [longest]
    assert seen[-1][0] == {"compute_timestamps": False}
    assert seen[-1][1:] == (1, 16)


def test_timestamps_batches_back_to_back_rebuild_nothing(mm, monkeypatch):
    model = FakeModel()
    instance = _instance(mm, monkeypatch, model)
    for _ in range(3):
        instance._infer_waveforms(model, ["w"], with_timestamps=True)
    assert not hasattr(model, "rebuilt_with")
    assert instance.warmups == []
    instance._infer_waveforms(model, ["w"], with_timestamps=False)
    instance._infer_waveforms(model, ["w"], with_timestamps=False)
    assert instance.warmups == [True]


def test_direct_batches_without_timestamps_never_rebuild(mm, monkeypatch):
    model = FakeModel()
    instance = _instance(mm, monkeypatch, model)
    warmed = model.decoding
    for _ in range(3):
        instance._infer_waveforms(model, ["w"], with_timestamps=False)
    assert model.decoding is warmed
    assert instance.warmups == []
    assert instance.graph_checks == []


def test_runtime_graph_mode_failure_logs_and_serves(mm, monkeypatch, caplog):
    model = FakeModel()
    instance = _instance(mm, monkeypatch, model, graph_mode="no_graphs")
    instance._infer_waveforms(model, ["w"], with_timestamps=True)
    with caplog.at_level("ERROR"):
        out = instance._infer_waveforms(model, ["w"], with_timestamps=False)
    assert out == ["direct 1"]
    assert "not in full_graph mode" in caplog.text
    # Not retried on every batch: the rebuilt object is the serving reference.
    instance._infer_waveforms(model, ["w"], with_timestamps=False)
    assert instance.warmups == [True]


def test_startup_restore_enforces_full_graph(mm, monkeypatch):
    model = FakeModel()
    instance = _instance(mm, monkeypatch, model, graph_mode="no_graphs")
    instance._check_transcribe_mode(model)
    with pytest.raises(RuntimeError, match="REQUIRE_FULL_CUDA_GRAPH"):
        instance._ensure_serving_decoding(model, phase="after_transcribe", startup=True)
    # Full warmup at startup, not decoder-only.
    assert instance.warmups == [False]


def test_restore_without_snapshot_keeps_decoding(mm):
    model = FakeModel()
    model.transcribe(["a"], timestamps=True)
    replaced = model.decoding
    mm._restore_eval_after_transcribe(model)
    assert model.decoding is replaced


def test_startup_mode_check_fails_if_training_mode_remains(mm, monkeypatch):
    model = FakeModel()
    instance = mm.Model()
    instance._model = model
    mm._snapshot_decoding(model)
    instance._check_transcribe_mode(model)  # restored: passes
    monkeypatch.setattr(mm, "_restore_eval_after_transcribe", lambda _model: [])
    with pytest.raises(RuntimeError, match="training mode"):
        instance._check_transcribe_mode(model)
