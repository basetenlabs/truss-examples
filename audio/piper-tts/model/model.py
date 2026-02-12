import base64
import os
import subprocess
from typing import Dict

OUTPUT_FILE_PATH = "output.wav"


class Model:
    def __init__(self, **kwargs):
        self._data_dir = kwargs["data_dir"]
        self.model_path = os.path.join(self._data_dir, "models", "model.onnx")

    def load(self):
        pass

    def wav_to_base64(self, file_path):
        with open(file_path, "rb") as wav_file:
            binary_data = wav_file.read()
            base64_data = base64.b64encode(binary_data)
            base64_string = base64_data.decode("utf-8")
            return base64_string

    def predict(self, model_input: Dict) -> Dict:
        prompt = model_input.get("text")
        command = f"echo '{prompt}' | piper --model {self.model_path} --cuda --output_file {OUTPUT_FILE_PATH}"
        subprocess.run(command, shell=True)
        b64_output = self.wav_to_base64(OUTPUT_FILE_PATH)
        os.remove(OUTPUT_FILE_PATH)
        return {"output": b64_output}
