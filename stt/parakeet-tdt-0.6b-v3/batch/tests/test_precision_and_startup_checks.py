"""PRECISION_MODE / INFERENCE_DTYPE, ENCODER_CUDNN, the startup checks and batch timing.

Not run by this repository's CI, which has no Python test job -- run by hand with

    uv run --with pytest --with numpy --with prometheus-client pytest stt/parakeet-tdt-0.6b-v3/batch/tests

NeMo and torch are stubbed (torch only for the full-precision cast test, which is
skipped when torch is not installed). The real casts, CUDA-graph capture and
transcript accuracy are exercised by the deployment's startup checks.
"""

from __future__ import annotations

import contextlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "model.py"
BASE_ENV = {
    "MAX_BATCH_SIZE": "16",
    "BATCH_WINDOW_MS": "5",
    "BATCH_BUCKET_SECONDS": "2,4,8",
    "PARAKEET_DIRECT_FORWARD": "true",
}
CONFIG_KEYS = (
    "PRECISION_MODE",
    "INFERENCE_DTYPE",
    "ENCODER_CUDNN",
    "BATCH_LENGTH_BUCKETING",
    "BATCH_BUCKET_LOOKAHEAD",
    "BATCH_MIN_PADDING_EFFICIENCY",
    "STARTUP_TRANSCRIPT_CHECK",
    "STARTUP_MODE_CHECK",
    "TIMING_LOG_BATCHES",
    "ENCODER_CUDA_GRAPHS",
    "ENCODER_GRAPH_BATCH_SIZES",
    "ENCODER_GRAPH_MAX_SECONDS",
    "ENCODER_GRAPH_STEP_SECONDS",
    "ENCODER_POS_TABLE_SECONDS",
    "GRAPH_PARITY_BATCHES",
)


def _load(monkeypatch, **env):
    from prometheus_client import REGISTRY

    for collector, names in list(REGISTRY._collector_to_names.items()):
        if any(name.startswith("parakeet_") for name in names):
            REGISTRY.unregister(collector)
    for key in CONFIG_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in {**BASE_ENV, **env}.items():
        monkeypatch.setenv(key, value)
    spec = importlib.util.spec_from_file_location("parakeet_tdt_precision", MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_defaults_are_round_two_winners(monkeypatch):
    mm = _load(monkeypatch)
    assert mm.PRECISION_MODE == "full"
    assert mm.INFERENCE_DTYPE == "bfloat16"
    assert mm.ENCODER_CUDNN is False
    assert mm.ENCODER_CUDA_GRAPHS is False
    assert mm.GRAPH_PARITY_BATCHES == 0
    assert mm.BATCH_LENGTH_BUCKETING is False
    assert mm.BATCH_BUCKET_LOOKAHEAD == 4 * mm.MAX_BATCH_SIZE
    assert mm.BATCH_MIN_PADDING_EFFICIENCY == 0
    assert mm.TIMING_LOG_BATCHES == 0
    assert mm.STARTUP_TRANSCRIPT_CHECK is True
    assert mm.STARTUP_MODE_CHECK is True


def test_encoder_cudnn_flag(monkeypatch):
    assert _load(monkeypatch, ENCODER_CUDNN="true").ENCODER_CUDNN is True
    assert _load(monkeypatch, ENCODER_CUDNN="false").ENCODER_CUDNN is False


def test_config_yaml_sets_serving_defaults():
    yaml = pytest.importorskip("yaml")
    config = yaml.safe_load((MODEL_PATH.parents[1] / "config.yaml").read_text())
    env = config["environment_variables"]
    assert env["ENCODER_CUDNN"] == "false"
    assert env["PRECISION_MODE"] == "full"
    assert env["INFERENCE_DTYPE"] == "bfloat16"
    assert env["ENCODER_CUDA_GRAPHS"] == "true"
    assert env["BATCH_WINDOW_EARLY_EXIT"] == "true"
    assert env["ENCODER_GRAPH_BATCH_SIZES"] == "1,2,4,6,8,10,12,14,16"


@pytest.mark.parametrize(
    "env",
    [
        {"PRECISION_MODE": "half"},
        {"INFERENCE_DTYPE": "fp8"},
        {"BATCH_BUCKET_LOOKAHEAD": "0"},
        {"BATCH_MIN_PADDING_EFFICIENCY": "1.5"},
        {"TIMING_LOG_BATCHES": "-1"},
        {"ENCODER_GRAPH_BATCH_SIZES": "0,4"},
        {"ENCODER_GRAPH_MAX_SECONDS": "0"},
        {"ENCODER_GRAPH_STEP_SECONDS": "-1"},
        {"GRAPH_PARITY_BATCHES": "-1"},
        # Graphs are captured in the serving dtype only, never under autocast.
        {"ENCODER_CUDA_GRAPHS": "true", "PRECISION_MODE": "autocast"},
        {"ENCODER_CUDA_GRAPHS": "true", "PARAKEET_DIRECT_FORWARD": "false"},
    ],
)
def test_invalid_config_is_rejected(monkeypatch, env):
    with pytest.raises(ValueError):
        _load(monkeypatch, **env)


@pytest.fixture
def fake_torch(monkeypatch):
    torch = types.ModuleType("torch")
    calls = []
    torch.float16, torch.bfloat16 = "float16", "bfloat16"

    def autocast(**kwargs):
        calls.append(kwargs)
        return contextlib.nullcontext()

    torch.autocast = autocast
    torch.inference_mode = contextlib.nullcontext
    monkeypatch.setitem(sys.modules, "torch", torch)
    return calls


def test_autocast_mode_uses_inference_dtype(monkeypatch, fake_torch):
    mm = _load(monkeypatch, PRECISION_MODE="autocast", INFERENCE_DTYPE="bfloat16")
    with mm._autocast():
        pass
    assert fake_torch == [{"device_type": "cuda", "dtype": "bfloat16"}]


@pytest.mark.parametrize(
    "env",
    [
        {"PRECISION_MODE": "full", "INFERENCE_DTYPE": "bfloat16"},
        {"PRECISION_MODE": "full", "INFERENCE_DTYPE": "float16"},
        {"PRECISION_MODE": "autocast", "INFERENCE_DTYPE": "float32"},
    ],
)
def test_full_mode_and_float32_never_autocast(monkeypatch, fake_torch, env):
    mm = _load(monkeypatch, **env)
    assert isinstance(mm._autocast(), contextlib.nullcontext)
    assert fake_torch == []


def test_word_error_rate_ignores_case_and_punctuation(monkeypatch):
    mm = _load(monkeypatch)
    # The checkpoint's punctuated, capitalized output for the startup clip.
    text = (
        "Well, I don't wish to see it any more, observed Phoebe, turning away her "
        "eyes. It is certainly very like the old portrait."
    )
    assert mm._word_error_rate(mm.STARTUP_CHECK_REFERENCE, text) == 0
    assert mm._word_error_rate(mm.STARTUP_CHECK_REFERENCE, "") == 1
    assert mm._word_error_rate("a b c d", "a x c d") == 0.25


def test_full_precision_casts_encoder_decoder_and_joint(monkeypatch):
    torch = pytest.importorskip("torch")
    mm = _load(monkeypatch, PRECISION_MODE="full", INFERENCE_DTYPE="bfloat16")

    class Preprocessor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("window", torch.ones(4))
            self.register_buffer(
                "dtype_sentinel_tensor", torch.tensor((), dtype=torch.float32), persistent=False
            )

    model = types.SimpleNamespace(
        preprocessor=Preprocessor(),
        encoder=torch.nn.Linear(4, 4),
        decoder=torch.nn.LSTM(4, 4),
        joint=torch.nn.Linear(4, 4),
    )
    mm.Model._cast_full_precision(model)
    for name in ("encoder", "decoder", "joint"):
        assert all(p.dtype == torch.bfloat16 for p in getattr(model, name).parameters()), name
    # The mel front end keeps float32 values; only its output dtype changes.
    assert model.preprocessor.window.dtype == torch.float32
    assert model.preprocessor.dtype_sentinel_tensor.dtype == torch.bfloat16


def test_full_precision_requires_the_dtype_sentinel(monkeypatch):
    torch = pytest.importorskip("torch")
    mm = _load(monkeypatch, PRECISION_MODE="full", INFERENCE_DTYPE="bfloat16")
    model = types.SimpleNamespace(
        preprocessor=torch.nn.Module(),
        encoder=torch.nn.Linear(4, 4),
        decoder=torch.nn.Linear(4, 4),
        joint=torch.nn.Linear(4, 4),
    )
    with pytest.raises(RuntimeError, match="dtype_sentinel_tensor"):
        mm.Model._cast_full_precision(model)


class _Module:
    def __init__(self, training=False):
        self.training = training


class _NamedModel:
    def __init__(self, training_names=()):
        self._modules = {
            "": _Module(),
            "encoder": _Module("encoder" in training_names),
            "encoder.layers.0.dropout": _Module("encoder.layers.0.dropout" in training_names),
            "joint": _Module("joint" in training_names),
        }

    def named_modules(self):
        return iter(self._modules.items())


def test_eval_mode_assertion(monkeypatch):
    mm = _load(monkeypatch)
    instance = mm.Model()
    instance._model = _NamedModel()
    instance._assert_eval_mode("startup")
    instance._model = _NamedModel({"encoder.layers.0.dropout"})
    with pytest.raises(RuntimeError, match="training mode"):
        instance._assert_eval_mode("startup")


def test_decoding_restored_and_rewarmed_after_startup_transcribe(monkeypatch):
    mm = _load(monkeypatch, REQUIRE_FULL_CUDA_GRAPH="true")
    instance = mm.Model()
    calls = []

    class Cfg(dict):
        pass

    def change_decoding_strategy(cfg, verbose):
        calls.append(("change", dict(cfg)))
        instance._model.cfg.decoding = cfg
        instance._model.decoding = object()

    instance._model = types.SimpleNamespace(
        cfg=types.SimpleNamespace(decoding=Cfg(compute_timestamps=True)),
        change_decoding_strategy=change_decoding_strategy,
    )
    instance._decoding_cfg = Cfg(compute_timestamps=None)
    monkeypatch.setattr(
        instance,
        "_warmup",
        lambda *, decoder_only: calls.append(("warmup", decoder_only)),
    )
    monkeypatch.setattr(
        instance,
        "_log_decoder_graph_mode",
        lambda phase, *, enforce: calls.append(("graph", phase, enforce)),
    )
    instance._restore_decoding("rewarm_after_transcribe")
    assert calls == [
        ("change", {"compute_timestamps": None}),
        ("warmup", False),
        ("graph", "rewarm_after_transcribe", True),
    ]
    assert instance._serving_decoding is instance._model.decoding
