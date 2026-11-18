import base64
import os
import tempfile
from tempfile import NamedTemporaryFile
from typing import Dict
import requests

from TTS.api import TTS

DEFAULT_SPEAKER_NAME = "Claribel Dervla"


class Model:
    def __init__(self, **kwargs):
        self.model = None

    def load(self):
        self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

    def base64_to_wav(self, base64_string, output_file_path):
        binary_data = base64.b64decode(base64_string)
        with open(output_file_path, "wb") as wav_file:
            wav_file.write(binary_data)

    def wav_to_base64(self, file_path):
        with open(file_path, "rb") as wav_file:
            binary_data = wav_file.read()
            base64_data = base64.b64encode(binary_data)
            base64_string = base64_data.decode("utf-8")
            return base64_string

    def download_audio(self, url: str) -> str:
        """Download audio from URL to a temporary file and return the path."""
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        fd, path = tempfile.mkstemp(suffix=".wav")
        try:
            os.write(fd, response.content)
        finally:
            os.close(fd)
        return path

    def predict(self, request: Dict) -> Dict:
        text = request.get("text")
        speaker_voice = request.get("speaker_voice", DEFAULT_SPEAKER_NAME)
        language = request.get("language", "en")
        speaker_wav = request.get("speaker_wav")
        if speaker_wav:
            # Download audio if it's a URL
            temp_speaker_path = None
            if speaker_wav.startswith(("http://", "https://")):
                temp_speaker_path = self.download_audio(speaker_wav)
                speaker_wav = temp_speaker_path
            try:
                with NamedTemporaryFile(delete=True) as fp:
                    self.model.tts_to_file(
                        text=text,
                        file_path=fp.name,
                        speaker_wav=speaker_wav,
                        language=language,
                    )
                    base64_string = self.wav_to_base64(fp.name)
                    return {"output": base64_string}
            finally:
                if temp_speaker_path and os.path.exists(temp_speaker_path):
                    os.remove(temp_speaker_path)
        with NamedTemporaryFile(delete=True) as fp:
            self.model.tts_to_file(
                text=text,
                file_path=fp.name,
                speaker=speaker_voice,
                language=language,
            )

            base64_string = self.wav_to_base64(fp.name)
            return {"output": base64_string}
