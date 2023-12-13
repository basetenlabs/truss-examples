import base64
import os
import subprocess
from typing import Dict

import numpy as np
import requests
import torch
from scipy.io.wavfile import write


def download_model(model_url, destination_path):
    print(f"Downloading model {model_url} ...")
    try:
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        print("download response: ", response)

        # Open the destination file and write the content in chunks
        print("opening: ", destination_path)
        with open(destination_path, "wb") as file:
            print("writing chunks...")
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive new chunks
                    file.write(chunk)

            print("done writing chunks!!!!")

        print(f"Downloaded file to: {destination_path}")
    except requests.exceptions.RequestException as e:
        print(f"Download failed: {e}")


pip_install_command = ["pip", "install", "-r", "/app/model/StyleTTS2/requirements.txt"]
subprocess.run(pip_install_command, check=True)


download_model(
    "https://huggingface.co/yl4579/StyleTTS2-LJSpeech/resolve/main/Models/LJSpeech/epoch_2nd_00100.pth",
    "./model/StyleTTS2/Models/LJSpeech/epoch_2nd_00100.pth",
)

download_model(
    "https://github.com/yl4579/StyleTTS2/raw/main/Utils/ASR/epoch_00080.pth",
    "./model/StyleTTS2/Utils/ASR/epoch_00080.pth",
)

download_model(
    "https://github.com/yl4579/StyleTTS2/raw/main/Utils/JDC/bst.t7",
    "./model/StyleTTS2/Utils/JDC/bst.t7",
)

download_model(
    "https://github.com/yl4579/StyleTTS2/raw/main/Utils/PLBERT/step_1000000.t7",
    "./model/StyleTTS2/Utils/PLBERT/step_1000000.t7",
)

from model.StyleTTS2.ljspeech_helper import lj_speech_inference

device = "cuda" if torch.cuda.is_available() else "cpu"
original_working_directory = os.getcwd()


class Model:
    def __init__(self, **kwargs):
        # self._data_dir = kwargs["data_dir"]
        pass

    def load(self):
        pass

    def wav_to_base64(self, file_path):
        with open(file_path, "rb") as wav_file:
            binary_data = wav_file.read()
            base64_data = base64.b64encode(binary_data)
            base64_string = base64_data.decode("utf-8")
            return base64_string

    def convert_np_arr_to_wav(self, wav):
        sample_rate = 24000
        scaled_audio_data = np.int16(wav * 32767)
        output_file_path = "output_wav_audio.wav"
        write(output_file_path, sample_rate, scaled_audio_data)
        return self.wav_to_base64(output_file_path)

    def predict(self, model_input: Dict) -> Dict:
        text = model_input.get("text")
        noise = torch.randn(1, 1, 256).to(device)
        wav = lj_speech_inference(text, noise, diffusion_steps=5, embedding_scale=1)
        b64_output = self.convert_np_arr_to_wav(wav)
        return {"output": b64_output}
