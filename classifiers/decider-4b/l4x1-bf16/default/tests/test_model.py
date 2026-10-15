import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from model.model import MAX_TOKENS, MODEL_NAME, MODEL_PATH, Model


@pytest.fixture
def runtime(monkeypatch):
    upstream = Mock()
    upstream.m.tok.encode.return_value = [1, 2]
    upstream.system_one.return_value = {
        'model': 'decider-4b-v2.1', 'answers': {'yes': {'type': 'noul', 'noul': 0.9}},
    }
    factory = Mock(return_value=upstream)
    render = Mock(return_value='rendered state')
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(bfloat16='bf16'))
    monkeypatch.setitem(sys.modules, 'decider.infer', SimpleNamespace(Decider=factory))
    monkeypatch.setitem(sys.modules, 'decider.systemone', SimpleNamespace(render_state=render))
    model = Model()
    model.load()
    upstream.system_one.reset_mock()
    return model, upstream, factory, render


def test_loader_preserves_upstream_calibration_defaults(runtime):
    _, _, factory, _ = runtime
    factory.assert_called_once_with(MODEL_PATH, device='cuda:0', dtype='bf16', use_graphs=False)


def test_preserves_state_questions_and_answer_probabilities(runtime):
    model, upstream, _, render = runtime
    state = {'ticket': 'damaged'}
    questions = {'yes': {'type': 'noul'}}
    result = model.predict({'model': MODEL_NAME, 'state': state, 'questions': questions})
    render.assert_called_with(state)
    upstream.m.tok.encode.assert_called_with('Context:\nrendered state', add_special_tokens=False)
    upstream.system_one.assert_called_once_with(state=state, questions=questions, max_state_tokens=MAX_TOKENS)
    assert result['answers'] is upstream.system_one.return_value['answers']
    assert result['model'] == MODEL_NAME


@pytest.mark.parametrize('token_count', [MAX_TOKENS - 1, MAX_TOKENS])
def test_context_at_or_below_limit_is_preserved(runtime, token_count):
    model, upstream, _, _ = runtime
    upstream.m.tok.encode.return_value = [0] * token_count
    model.predict({'model': MODEL_NAME, 'state': 'state', 'questions': {'q': {'type': 'noul'}}})
    upstream.system_one.assert_called_once()


def test_oversized_context_is_rejected_before_upstream_truncation(runtime):
    model, upstream, _, _ = runtime
    upstream.m.tok.encode.return_value = [0] * (MAX_TOKENS + 1)
    with pytest.raises(ValueError, match='context exceeds'):
        model.predict({'model': MODEL_NAME, 'state': 'long', 'questions': {'q': {'type': 'noul'}}})
    upstream.system_one.assert_not_called()


def test_prefix_can_push_otherwise_fitting_state_over_limit(runtime):
    model, upstream, _, render = runtime
    state = 'x' * MAX_TOKENS
    render.return_value = state
    # A character tokenizer isolates the prefix accounting from model inference.
    upstream.m.tok.encode.side_effect = lambda text, **_: list(text)
    with pytest.raises(ValueError, match='context exceeds'):
        model.predict({'model': MODEL_NAME, 'state': state, 'questions': {'q': {'type': 'noul'}}})
    upstream.system_one.assert_not_called()
