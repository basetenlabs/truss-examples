import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory


@contextmanager
def checkpoint_package(mount):
    """Give upstream its exact bundle, without platform mount metadata or links.

    Upstream still verifies every file's hash and the loaded parameter counts.
    Hard links avoid duplicating weights when the mount allows them; otherwise
    copy into the writable temporary directory.
    """
    mount = Path(mount)
    manifest = json.loads((mount / "MODEL_MANIFEST.json").read_text())
    with TemporaryDirectory(prefix="kai-") as directory:
        package = Path(directory)
        for name in ["MODEL_MANIFEST.json", *manifest["files_sha256"]]:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe checkpoint path: {name}")
            source = (mount / relative).resolve(strict=True)
            target = package / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, target)
            except OSError:
                shutil.copyfile(source, target)
        yield package


class Model:
    def __init__(self, **kwargs):
        self._model = None

    def load(self):
        from transformers import AutoModel

        # Keep mount metadata outside the upstream bundle's exact inventory.
        with checkpoint_package("/models/kai") as package:
            self._model = AutoModel.from_pretrained(
                str(package),
                trust_remote_code=True,
                local_files_only=True,
                device="cuda:0",
            )
        self._model.system_one(
            state="The parcel arrived damaged.",
            questions={
                "damaged": {
                    "type": "noul",
                    "instructions": "Is the parcel damaged?",
                }
            },
        )

    def predict(self, model_input):
        return self._model.system_one(
            state=model_input["state"], questions=model_input["questions"]
        )
