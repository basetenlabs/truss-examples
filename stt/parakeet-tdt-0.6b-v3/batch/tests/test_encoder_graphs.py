"""Shape selection and padding for ENCODER_CUDA_GRAPHS in model/model.py.

Not run by this repository's CI, which has no Python test job -- run by hand with

    uv run --with pytest --with numpy --with prometheus-client pytest stt/parakeet-tdt-0.6b-v3/batch/tests

CPU only: these cover which captured (batch, length) shape a batch replays, which
batches run eagerly, how a batch is zero-padded into the graph's static input, and
the load-order hooks around capture. Capture and replay themselves need a GPU and
are exercised by the deployment's startup checks (capture failure and a startup
transcript that does not run through a graph both fail load()).
"""

from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import numpy as np
import pytest

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "model.py"
SR = 16_000
CONFIG_KEYS = (
    "PRECISION_MODE",
    "INFERENCE_DTYPE",
    "ENCODER_CUDNN",
    "ENCODER_CUDA_GRAPHS",
    "ENCODER_GRAPH_BATCH_SIZES",
    "ENCODER_GRAPH_MAX_SECONDS",
    "ENCODER_GRAPH_STEP_SECONDS",
    "ENCODER_POS_TABLE_SECONDS",
    "GRAPH_PARITY_BATCHES",
    "WARMUP_AUDIO_SECONDS",
    "WARMUP_BATCH_SIZE",
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
    spec = importlib.util.spec_from_file_location("parakeet_tdt_graphs", MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def mm(monkeypatch):
    return _load(monkeypatch)


def test_graph_env_defaults(mm):
    assert mm.ENCODER_CUDA_GRAPHS is False
    assert mm.ENCODER_GRAPH_BATCH_SIZES == (1, 2, 4, 8, 16)
    assert mm.ENCODER_GRAPH_MAX_SECONDS == 30
    assert mm.ENCODER_GRAPH_STEP_SECONDS == 1
    assert mm.ENCODER_POS_TABLE_SECONDS == 1440


def test_graphs_allowed_in_full_precision(monkeypatch):
    mm = _load(monkeypatch, ENCODER_CUDA_GRAPHS="true", PRECISION_MODE="full")
    assert mm.ENCODER_CUDA_GRAPHS is True
    mm = _load(
        monkeypatch,
        ENCODER_CUDA_GRAPHS="true",
        PRECISION_MODE="autocast",
        INFERENCE_DTYPE="float32",
    )
    assert mm.ENCODER_CUDA_GRAPHS is True


@pytest.mark.parametrize(
    "sizes, max_batch, expected",
    [
        ((1, 2, 4, 8, 16), 16, (1, 2, 4, 8, 16)),
        ((1, 2, 4, 8, 16), 8, (1, 2, 4, 8)),
        ((1, 2, 4, 8, 16), 6, (1, 2, 4, 8)),
        ((1, 2, 4, 8, 16), 32, (1, 2, 4, 8, 16)),
        ((16, 1, 4, 4), 1, (1,)),
    ],
)
def test_graph_batch_sizes_stop_at_the_first_covering_size(mm, sizes, max_batch, expected):
    assert mm._graph_batch_sizes(sizes, max_batch) == expected


def test_default_capture_count_is_150(mm):
    sizes = mm._graph_batch_sizes(mm.ENCODER_GRAPH_BATCH_SIZES, 16)
    shapes = mm._graph_shapes(sizes, SR, 30 * SR)
    assert len(shapes) == 150
    assert len(set(shapes)) == 150
    # Largest first, so the shared pool peaks on the first capture.
    assert shapes[0] == (16, 30 * SR)
    assert shapes[-1] == (1, SR)


@pytest.mark.parametrize(
    "samples, padded",
    [(1, SR), (SR - 1, SR), (SR, SR), (SR + 1, 2 * SR), (29 * SR + 5, 30 * SR)],
)
def test_round_up_to_the_next_step(mm, samples, padded):
    assert mm._round_up_samples(samples, SR) == padded


@pytest.mark.parametrize(
    "batch, seconds, expected",
    [
        (1, 0.15, (1, SR)),
        (1, 1.0, (1, SR)),
        (2, 3.2, (2, 4 * SR)),
        (3, 2.0, (4, 2 * SR)),
        (5, 7.5, (8, 8 * SR)),
        (9, 0.4, (16, SR)),
        (16, 30.0, (16, 30 * SR)),
    ],
)
def test_graph_key_pads_batch_and_length_up(mm, batch, seconds, expected):
    sizes = (1, 2, 4, 8, 16)
    assert mm._graph_key(batch, round(seconds * SR), sizes, SR, 30 * SR) == expected


@pytest.mark.parametrize(
    "batch, samples",
    [
        (17, SR),  # larger than any captured batch
        (1, 30 * SR + 1),  # longer than the longest graph
        (4, 600 * SR),
        (0, SR),
        (1, 0),
    ],
)
def test_outside_the_captured_range_runs_eagerly(mm, batch, samples):
    assert mm._graph_key(batch, samples, (1, 2, 4, 8, 16), SR, 30 * SR) is None


def test_every_key_in_range_is_a_captured_shape(mm):
    sizes = (1, 2, 4, 8, 16)
    captured = set(mm._graph_shapes(sizes, SR, 30 * SR))
    for batch in range(1, 17):
        for samples in range(1, 30 * SR + 1, 7919):
            assert mm._graph_key(batch, samples, sizes, SR, 30 * SR) in captured


def test_non_integral_max_seconds_rounds_up(mm):
    graphs = mm.EncoderGraphs(None, (1, 2), SR // 2, round(7.2 * SR))
    assert graphs.max_padded == round(7.5 * SR)
    assert graphs.key(2, round(7.4 * SR)) == (2, round(7.5 * SR))
    assert graphs.key(2, round(7.6 * SR)) is None


def test_stack_waveforms_pads_rows_and_lengths(mm):
    waveforms = [np.ones(3, np.float32), np.full(5, 2, np.float32)]
    signals, lengths = mm._stack_waveforms(waveforms, 8, rows=4)
    assert signals.shape == (4, 8) and signals.dtype == np.float32
    assert lengths.dtype == np.int64
    np.testing.assert_array_equal(signals[0], [1, 1, 1, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(signals[1], [2, 2, 2, 2, 2, 0, 0, 0])
    # Batch-padding rows are silent clips of the full length, never empty clips.
    assert not signals[2:].any()
    np.testing.assert_array_equal(lengths, [3, 5, 8, 8])


def test_stack_waveforms_eager_shape_matches_previous_padding(mm):
    waveforms = [np.ones(3, np.float32), np.ones(5, np.float32)]
    signals, lengths = mm._stack_waveforms(waveforms, 5)
    assert signals.shape == (2, 5)
    np.testing.assert_array_equal(lengths, [3, 5])


@pytest.mark.parametrize("rows, length", [(1, 8), (2, 4)])
def test_stack_waveforms_rejects_what_does_not_fit(mm, rows, length):
    with pytest.raises(ValueError):
        mm._stack_waveforms([np.ones(5, np.float32)] * 2, length, rows=rows)


def test_disabled_graphs_return_none(mm):
    graphs = mm.EncoderGraphs(None, (1, 2), SR, 2 * SR)
    graphs.disable("test")
    graphs.disable("second reason is ignored")
    assert graphs.disabled_reason == "test"
    assert graphs.run([np.ones(SR, np.float32)]) is None


def test_encode_prefers_the_graph_and_falls_back_to_eager(mm):
    instance = mm.Model()
    instance._eager_encode = lambda waveforms: ("eager-out", "eager-len")
    assert instance._encode([np.ones(4, np.float32)]) == ("eager-out", "eager-len", "eager")

    class Graphs:
        def __init__(self, result):
            self.result = result

        def run(self, waveforms):
            return self.result

    instance._graphs = Graphs(("graph-out", "graph-len"))
    assert instance._encode([np.ones(4, np.float32)]) == ("graph-out", "graph-len", "graph")
    instance._graphs = Graphs(None)
    assert instance._encode([np.ones(4, np.float32)])[2] == "eager"


def test_warmup_sizes_the_decoder_for_the_largest_graph(monkeypatch):
    mm = _load(monkeypatch, WARMUP_AUDIO_SECONDS="0.5")
    instance = mm.Model()
    calls = []
    instance._infer_waveforms = lambda waveforms, *, with_timestamps: calls.append(
        (len(waveforms), waveforms[0].shape[0], with_timestamps)
    )
    instance._graphs = types.SimpleNamespace(batch_sizes=(1, 2, 4, 8, 16), max_padded=30 * SR)
    instance._warmup()
    assert calls == [(16, 30 * SR, False), (1, SR // 2, False)]

    calls.clear()
    instance._graphs = None
    instance._warmup()
    assert calls == [(1, SR // 2, False)]


def test_word_edits(mm):
    assert mm._word_edits("a b c", "a b c") == 0
    assert mm._word_edits("a b c", "a x c") == 1
    assert mm._word_edits("a b c", "a b") == 1
