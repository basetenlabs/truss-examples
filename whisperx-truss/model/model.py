import whisperx
import requests
import os
from typing import Dict


class Model:
    device = "cuda"
    batch_size = 16
    compute_type = "float16"

    def __init__(self, **kwargs):
        self._data_dir = kwargs["data_dir"]
        self.model = None

    def load(self):
        # Need to manually download vad model
        vad_model_path = os.path.join(self._data_dir, "models", "pytorch_model.bin")
        self.model = whisperx.load_model("medium", self.device, language="en", compute_type=self.compute_type, vad_options={"model_fp": vad_model_path})

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



