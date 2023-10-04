import whisperx
import gc
import requests
import os
import torch
import shutil
from typing import Dict


class Model:
    device = "cuda"
    batch_size = 16
    compute_type = "float16"

    def __init__(self, **kwargs):
        self.model = None

    def load(self):
        self.model = whisperx.load_model("large-v2", self.device, compute_type=self.compute_type)

    def predict(self, request: Dict) -> Dict:
        file = request.get("audio_file", None)
        if not file:
            raise Exception("An audio file is required for this model")

        res = requests.get(file)
        with open('audio.mp3', 'wb') as audio_file:
            audio_file.write(res.content)
            audio = whisperx.load_audio("audio.mp3")

            result = self.model.transcribe(audio, batch_size=self.batch_size)

            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
            result = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)

            segments_output = []
            for segment in result["segments"]:
                segments_output.append({"start": segment["start"], "end": segment["end"], "text": segment["text"]})

        os.remove("audio.mp3")

        return {"model_output": segments_output}



