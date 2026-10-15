import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from model.model import MAX_TOKENS, MODEL_NAME, MODEL_PATH, Model


@pytest.fixture
def runtime(monkeypatch):
    processor = SimpleNamespace(tokenizer=object())
    upstream = SimpleNamespace(
        load_release_model=Mock(return_value=(object(), processor)),
        encode_record=Mock(return_value=SimpleNamespace(input_ids=[1, 2])),
        systemone=Mock(return_value={'model': MODEL_NAME, 'answers': {'yes': {'type': 'noul', 'noul': 0.9}}}),
    )
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(bfloat16='bf16'))
    monkeypatch.setitem(sys.modules, 'joint_schema_model', upstream)
    model = Model()
    model.load()
    upstream.systemone.reset_mock()
    return model, upstream


def test_loader_uses_pinned_local_mount_and_bf16(runtime):
    _, upstream = runtime
    upstream.load_release_model.assert_called_once_with(
        MODEL_PATH, device='cuda:0', dtype='bf16', local_files_only=True,
    )


def test_preserves_payload_and_upstream_answers(runtime):
    model, upstream = runtime
    body = {'model': MODEL_NAME, 'state': {'ticket': 'broken'},
            'questions': {'yes': {'type': 'noul', 'instructions': None}}}
    result = model.predict(body)
    upstream.systemone.assert_called_once_with(model._model, model._processor, body, max_length=MAX_TOKENS)
    assert result == upstream.systemone.return_value


def test_oversized_input_is_rejected_before_upstream_truncation(runtime):
    model, upstream = runtime
    upstream.encode_record.return_value = SimpleNamespace(input_ids=[0] * (MAX_TOKENS + 1))
    with pytest.raises(ValueError, match='encoded tokens'):
        model.predict({'model': MODEL_NAME, 'state': 'long', 'questions': {'q': {'type': 'noul'}}})
    upstream.systemone.assert_not_called()
