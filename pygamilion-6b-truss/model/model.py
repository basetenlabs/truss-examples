from typing import Any

# Use a pipeline as a high-level helper
from transformers import pipeline


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self._pipe = None

    def load(self):
        self._pipe = pipeline("conversational", model="PygmalionAI/pygmalion-6b")

    def predict(self, model_input: Any) -> Any:
        return self._pipe(model_input.pop("prompt"), **model_input)
    o w
