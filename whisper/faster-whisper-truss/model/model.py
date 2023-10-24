from typing import Any, Dict
from tempfile import NamedTemporaryFile
import requests

from faster_whisper import WhisperModel


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self._model = None

    def load(self):
        self._model = WhisperModel("large-v2")

    def preprocess(self, request: Dict) -> Dict:
        resp = requests.get(request["url"])
        return {"response": resp.content}

    def predict(self, request: Dict) -> Dict:
        result_segments = []
        with NamedTemporaryFile() as fp:
            fp.write(request["response"])
            segments, info = self._model.transcribe(
                fp.name,
                temperature=0,
                best_of=5,
                beam_size=5,
            )

            for seg in segments:
                result_segments.append(
                    {"text": seg.text, "start": seg.start, "end": seg.end}
                )

        return {
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "segments": result_segments,
        }
