import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from model import model as adapter
from model.model import MODEL_NAME, MODEL_PATH, ONEJEV_PATH, Model


class FakeResponse:
    def __init__(self):
        self.answers = {"yes": {"type": "noul", "noul": 0.9}}

    def model_dump(self):
        return {"model": "", "answers": self.answers, "usage": {"input_tokens": 12, "output_tokens": 1}}


@pytest.fixture
def git(monkeypatch):
    run = Mock(return_value=SimpleNamespace(returncode=1))
    monkeypatch.setattr(adapter.subprocess, "run", run)
    return run


@pytest.fixture
def runtime(monkeypatch, git):
    engine = Mock()
    engine.info = SimpleNamespace(fork_mode="batched")
    engine.decide.return_value = (FakeResponse(), {})
    engine_cls = Mock(return_value=engine)
    request_cls = Mock(side_effect=lambda **kwargs: kwargs)
    monkeypatch.setitem(sys.modules, "qev", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "qev.engine", SimpleNamespace(DecisionEngine=engine_cls))
    monkeypatch.setitem(sys.modules, "qev.schema", SimpleNamespace(SystemOneRequest=request_cls))
    model = Model()
    model.load()
    engine.decide.reset_mock()
    return model, engine, engine_cls


def test_loader_patches_onejev_from_pinned_mount(runtime, git):
    _, _, engine_cls = runtime
    patch = f"{MODEL_PATH}/serve.patch"
    assert git.call_args_list[0].args[0] == ["git", "-C", ONEJEV_PATH, "apply", "--reverse", "--check", patch]
    assert git.call_args_list[1].args[0] == ["git", "-C", ONEJEV_PATH, "apply", patch]
    engine_cls.assert_called_once_with(MODEL_PATH, device="cuda:0")


def test_already_applied_patch_is_not_reapplied(git):
    git.return_value = SimpleNamespace(returncode=0)
    adapter.apply_serve_patch()
    assert git.call_count == 1


def test_preserves_payload_and_upstream_answers(runtime):
    model, engine, _ = runtime
    body = {"model": MODEL_NAME, "state": {"ticket": "broken"},
            "questions": {"yes": {"type": "noul", "instructions": ""}}}
    result = model.predict(body)
    engine.decide.assert_called_once_with({"state": body["state"], "questions": body["questions"]})
    assert result == {"model": MODEL_NAME, "answers": {"yes": {"type": "noul", "noul": 0.9}},
                      "usage": {"input_tokens": 12, "output_tokens": 1}}


def test_over_limit_request_is_refused(runtime):
    model, engine, _ = runtime
    engine.decide.side_effect = ValueError("request is 70000 tokens, limit is 65536")
    with pytest.raises(ValueError, match="limit is"):
        model.predict({"model": MODEL_NAME, "state": "long", "questions": {"q": {"type": "noul"}}})


def test_index_overflow_retries_sequentially_then_restores_mode(runtime):
    model, engine, _ = runtime
    modes = []

    def decide(request):
        modes.append(engine.info.fork_mode)
        if len(modes) == 1:
            raise RuntimeError("canUse32BitIndexMath(self) INTERNAL ASSERT FAILED")
        return FakeResponse(), {}

    engine.decide.side_effect = decide
    model.predict({"model": MODEL_NAME, "state": "s", "questions": {"q": {"type": "noul"}}})
    assert modes == ["batched", "sequential"]
    assert engine.info.fork_mode == "batched"


def test_other_runtime_errors_propagate(runtime):
    model, engine, _ = runtime
    engine.decide.side_effect = RuntimeError("CUDA error")
    with pytest.raises(RuntimeError, match="CUDA error"):
        model.predict({"model": MODEL_NAME, "state": "s", "questions": {"q": {"type": "noul"}}})
