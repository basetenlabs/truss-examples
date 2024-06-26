import base64
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict

import requests
import torch

import whisper


class Model:
    def __init__(self, **kwargs):
        self._data_dir = kwargs["data_dir"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None

    def load(self):
        self.model = whisper.load_model(
            Path(str(self._data_dir)) / "weights" / "large-v3.pt", self.device
        )

    def base64_to_wav(self, base64_string, output_file_path):
        binary_data = base64.b64decode(base64_string)
        with open(output_file_path, "wb") as wav_file:
            wav_file.write(binary_data)
        return output_file_path

    def predict(self, request: Dict) -> Dict:
        with NamedTemporaryFile() as fp:
            self.base64_to_wav(request["audio"], fp.name)
            result = whisper.transcribe(self.model, fp.name, temperature=0)
            segments = [
                {"start": r["start"], "end": r["end"], "text": r["text"]}
                for r in result["segments"]
            ]
        return {
            "language": whisper.tokenizer.LANGUAGES[result["language"]],
            "segments": segments,
            "text": result["text"],
        }
