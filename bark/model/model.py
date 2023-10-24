import base64
import io
from typing import Any

from scipy.io.wavfile import write

from bark import SAMPLE_RATE, generate_audio, preload_models


class Model:
    def load(self):
        preload_models()

    def predict(self, text_prompt: Any) -> Any:
        audio_array = generate_audio(text_prompt)
        return arr_to_b64(audio_array)


def arr_to_b64(arr):
    bytes_wav = bytes()
    byte_io = io.BytesIO(bytes_wav)
    write(byte_io, SAMPLE_RATE, arr)
    wav_bytes = byte_io.read()
    audio_data = base64.b64encode(wav_bytes).decode("UTF-8")
    return audio_data
