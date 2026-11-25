"""data/encoder_flex.py against stand-in NeMo modules (no torch or NeMo needed)."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "data" / "encoder_flex.py"


@pytest.fixture
def flex(monkeypatch):
    calls = []

    def flex_attention(*args, kernel_options=None, **kwargs):
        calls.append(kernel_options)
        return "out"

    utils = types.ModuleType("transformer_encoder_utils")
    utils._get_flex_attention = lambda x: flex_attention
    monkeypatch.setitem(sys.modules, "nemo.collections.asr.modules.transformer_encoder_utils", utils)
    spec = importlib.util.spec_from_file_location("encoder_flex_under_test", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, utils, calls


def test_forces_triton_and_keeps_other_options(flex):
    mod, utils, calls = flex
    mod.force_triton()
    attn = utils._get_flex_attention(None)
    assert attn("q", "k", "v") == "out"
    assert attn("q", "k", "v", kernel_options={"BLOCK_M": 64}) == "out"
    assert calls == [{"BACKEND": "TRITON"}, {"BLOCK_M": 64, "BACKEND": "TRITON"}]


def test_wraps_once(flex):
    mod, utils, calls = flex
    mod.force_triton()
    wrapped = utils._get_flex_attention
    mod.force_triton()
    assert utils._get_flex_attention is wrapped


def test_install_patches_when_target_already_imported(flex, monkeypatch):
    mod, utils, calls = flex
    monkeypatch.setitem(sys.modules, mod.TARGET, types.ModuleType("encoder_graphs"))
    mod.install()
    utils._get_flex_attention(None)("q")
    assert calls == [{"BACKEND": "TRITON"}]


def test_install_waits_for_target(flex, monkeypatch):
    mod, utils, _ = flex
    monkeypatch.delitem(sys.modules, mod.TARGET, raising=False)
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    mod.install()
    mod.install()
    assert sum(isinstance(f, mod._Finder) for f in sys.meta_path) == 1
    assert not getattr(utils, "_encoder_flex_wrapped", False)
