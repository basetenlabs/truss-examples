import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from model.model import Model


def fake_runtime():
    return SimpleNamespace(
        image_unavailable=None,
        video_unavailable=None,
        load_images=Mock(return_value=[object()]),
        load_videos=Mock(return_value=[object()]),
    )


def test_loader_uses_offline_mount_and_verifies_every_hash(monkeypatch):
    upstream = Mock(runtime=fake_runtime())
    auto_model = Mock()
    auto_model.from_pretrained.return_value = upstream
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModel=auto_model))
    model = Model()
    model.load()
    auto_model.from_pretrained.assert_called_once_with(
        "/models/d3", trust_remote_code=True, local_files_only=True,
        device="cuda:0", batch_size=1, verify="full",
    )
    assert upstream.system_one.call_args.kwargs["questions"]["damaged"]["type"] == "noul"


@pytest.mark.parametrize("kind", ["image", "video"])
def test_missing_processor_prevents_readiness(monkeypatch, kind):
    runtime = fake_runtime()
    setattr(runtime, f"{kind}_unavailable", "processor failed to load")
    upstream = Mock(runtime=runtime)
    auto_model = Mock()
    auto_model.from_pretrained.return_value = upstream
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModel=auto_model))
    with pytest.raises(RuntimeError, match=f"D3 {kind} inputs failed"):
        Model().load()
    upstream.system_one.assert_not_called()


def test_media_is_strictly_decoded_before_inference():
    runtime = fake_runtime()
    model = Model()
    model._model = Mock(runtime=runtime)
    images = ["data:image/png;base64,aW1hZ2U="]
    videos = ["data:video/mp4;base64,dmlkZW8="]
    questions = {"visible": {"type": "noul"}}
    model.predict({"questions": questions, "images": images, "videos": videos})
    runtime.load_images.assert_called_once_with(images, strict=True)
    runtime.load_videos.assert_called_once_with(videos, strict=True)
    model._model.system_one.assert_called_once_with(
        state=None, questions=questions,
        images=runtime.load_images.return_value, videos=runtime.load_videos.return_value,
    )


def test_invalid_media_does_not_reach_inference():
    runtime = fake_runtime()
    runtime.load_images.side_effect = ValueError("invalid base64")
    model = Model()
    model._model = Mock(runtime=runtime)
    with pytest.raises(ValueError, match="invalid base64"):
        model.predict({"questions": {"visible": {"type": "noul"}}, "images": ["bad"]})
    model._model.system_one.assert_not_called()


def test_text_only_keeps_media_absent():
    runtime = fake_runtime()
    runtime.load_images.return_value = []
    runtime.load_videos.return_value = []
    model = Model()
    model._model = Mock(runtime=runtime)
    questions = {"urgent": {"type": "noul"}}
    model.predict({"state": "Help today", "questions": questions})
    model._model.system_one.assert_called_once_with(
        state="Help today", questions=questions, images=None, videos=None,
    )
