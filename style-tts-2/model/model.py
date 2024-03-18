import base64
import os
from typing import Dict

import numpy as np
import torch
from scipy.io.wavfile import write
from StyleTTS2.model_download_helper import download_all_models

download_all_models()
from StyleTTS2.helper import compute_style, inference
from StyleTTS2.ljspeech_helper import lj_speech_inference

DEFAULT_DIFFUSION_STEPS = 5
DEFAULT_EMBEDDING_SCALE = 1
DEFAULT_ALPHA = 0.3
DEFAULT_BETA = 0.7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Model:
    def __init__(self, **kwargs):
        # self._data_dir = kwargs["data_dir"]
        self.model = None

    def load(self):
        pass

    def wav_to_base64(self, file_path):
        with open(file_path, "rb") as wav_file:
            binary_data = wav_file.read()
            base64_data = base64.b64encode(binary_data)
            base64_string = base64_data.decode("utf-8")
            return base64_string

    def base64_to_wav(self, base64_string, output_file_path):
        binary_data = base64.b64decode(base64_string)
        with open(output_file_path, "wb") as wav_file:
            wav_file.write(binary_data)
        return output_file_path

    def convert_np_arr_to_wav(self, wav):
        sample_rate = 24000
        scaled_audio_data = np.int16(wav * 32767)
        output_file_path = "output_wav_audio.wav"
        write(output_file_path, sample_rate, scaled_audio_data)
        return self.wav_to_base64(output_file_path)

    def predict(self, model_input: Dict) -> Dict:
        text = model_input.get("text")
        reference_audio = model_input.get("reference_audio", None)
        diffusion_steps = model_input.get("diffusion_steps", DEFAULT_DIFFUSION_STEPS)
        embedding_scale = model_input.get("embedding_scale", DEFAULT_EMBEDDING_SCALE)
        if reference_audio:
            reference_audio_path = self.base64_to_wav(
                reference_audio, "reference_audio_file.wav"
            )
            alpha = model_input.get("alpha", DEFAULT_ALPHA)
            beta = model_input.get("beta", DEFAULT_BETA)
            wav = inference(
                text,
                compute_style(reference_audio_path),
                alpha=alpha,
                beta=beta,
                diffusion_steps=diffusion_steps,
                embedding_scale=embedding_scale,
            )
            os.remove(reference_audio_path)
        else:
            noise = torch.randn(1, 1, 256).to(DEVICE)
            wav = lj_speech_inference(
                text,
                noise,
                diffusion_steps=diffusion_steps,
                embedding_scale=embedding_scale,
            )
        b64_output = self.convert_np_arr_to_wav(wav)
        return {"output": b64_output}
