from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict

import ffmpeg
import numpy as np
import requests
import torch

import whisper


class Model:
    def __init__(self, **kwargs):
        self._data_dir = kwargs["data_dir"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None

    def audio_url_to_waveform(self, path_or_url: str):
        sampling_rate = 16000
        # Use ffmpeg to read the audio file and convert to the monochannel, 16kHz
        out, _ = (
            ffmpeg.input(
                path_or_url, seekable=0
            )  # Disable HTTP seekable (range requests)
            .output("pipe:", format="wav", acodec="pcm_s16le", ac=1, ar=sampling_rate)
            .run(capture_stdout=True, capture_stderr=True)
        )

        # Convert the raw byte data into a numpy array
        waveform_np = np.frombuffer(out, dtype=np.int16)

        # Normalize the waveform data
        waveform_np = waveform_np.astype(np.float32) / 32768.0

        # Convert the numpy array to a pytorch tensor
        waveform_tensor = torch.tensor(waveform_np, dtype=torch.float32)

        return waveform_tensor

    def load(self):
        self.model = whisper.load_model(
            Path(str(self._data_dir)) / "weights" / "large-v3.pt", self.device
        )

    def predict(self, request: Dict) -> Dict:
        url = request.get("url")
        temperature = request.get("temperature", 0)

        waveform = self.audio_url_to_waveform(url)

        result = whisper.transcribe(self.model, waveform, temperature=temperature)
        segments = [
            {"start": r["start"], "end": r["end"], "text": r["text"]}
            for r in result["segments"]
        ]
        return {
            "language": whisper.tokenizer.LANGUAGES[result["language"]],
            "segments": segments,
            "text": result["text"],
        }
