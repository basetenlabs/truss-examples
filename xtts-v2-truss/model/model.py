import base64
import os
from typing import Dict

from TTS.api import TTS


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

    def preprocess(self, request: Dict) -> Dict:
        text = request.get("text")
        speaker_voice = request.get("speaker_voice")
        language = request.get("language")
        supported_languages = {
            "en",
            "es",
            "fr",
            "de",
            "it",
            "pt",
            "pl",
            "tr",
            "ru",
            "nl",
            "cs",
            "ar",
            "zh-cn",
        }

        if language not in supported_languages:
            return {
                "output": f"The language you chose is not supported. Please select from the following choices: {supported_languages}"
            }

        self.base64_to_wav(speaker_voice, "speaker_voice.wav")
        return {
            "text": text,
            "speaker_voice": "speaker_voice.wav",
            "language": language,
        }

    def predict(self, request: Dict) -> Dict:
        text = request.pop("text")
        speaker_voice = request.pop("speaker_voice")
        language = request.pop("language")
        self.model.tts_to_file(
            text=text,
            file_path="output.wav",
            speaker_wav=speaker_voice,
            language=language,
        )

        base64_string = self.wav_to_base64("output.wav")

        os.remove("speaker_voice.wav")
        os.remove("output.wav")
        return {"output": base64_string}
