import hashlib
import json
from pathlib import Path

import pytest
from model.model import checkpoint_package


def bundle(tmp_path):
    mount = tmp_path / "mount"
    mount.mkdir()
    target = tmp_path / "weights"
    target.write_bytes(b"weights")
    (mount / "backbone").mkdir()
    (mount / "backbone/model.safetensors").symlink_to(target)
    manifest = {"files_sha256": {
        "backbone/model.safetensors": hashlib.sha256(target.read_bytes()).hexdigest()
    }}
    (mount / "MODEL_MANIFEST.json").write_text(json.dumps(manifest))
    (mount / "platform-metadata").write_text("not part of the checkpoint")
    return mount, manifest


@pytest.mark.parametrize("copy_fallback", [False, True])
def test_package_preserves_manifest_files_and_excludes_mount_metadata(tmp_path, monkeypatch, copy_fallback):
    mount, manifest = bundle(tmp_path)
    if copy_fallback:
        def unavailable(*args):
            raise OSError("Cross-device link")
        monkeypatch.setattr("model.model.os.link", unavailable)
    with checkpoint_package(mount) as package:
        assert {str(p.relative_to(package)) for p in package.rglob("*") if p.is_file()} == {
            "MODEL_MANIFEST.json", *manifest["files_sha256"]
        }
        weight = package / "backbone/model.safetensors"
        assert not weight.is_symlink()
        assert hashlib.sha256(weight.read_bytes()).hexdigest() == manifest["files_sha256"]["backbone/model.safetensors"]
        assert (mount / "platform-metadata").exists()
    assert not package.exists()


def test_missing_manifest_file_fails_before_loading(tmp_path):
    mount, _ = bundle(tmp_path)
    (mount / "backbone/model.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        with checkpoint_package(mount):
            pytest.fail("An incomplete bundle must not reach model loading")
