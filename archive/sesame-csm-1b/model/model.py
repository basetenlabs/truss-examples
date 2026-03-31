import base64
from io import BytesIO
from huggingface_hub import hf_hub_download
from generator import load_csm_1b
import torchaudio


class Model:
    def __init__(self, **kwargs):
        self.generator = None
        self._secrets = kwargs["secrets"]

    def load(self):
        model_path = hf_hub_download(
            repo_id="sesame/csm-1b",
            filename="ckpt.pt",
            token=self._secrets["hf_access_token"],
        )
        self.generator = load_csm_1b(
            model_path, "cuda", self._secrets["hf_access_token"]
        )

    def wav_to_base64(self, wav_tensor):
        buffer = BytesIO()
        torchaudio.save(
            buffer,
            wav_tensor.unsqueeze(0).cpu(),
            self.generator.sample_rate,
            format="wav",
        )
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    def predict(self, model_input):
        text = model_input.get("text", "Hello from Sesame.")
        speaker = model_input.get("speaker", 0)
        audio = self.generator.generate(
            text=text,
            speaker=speaker,
            context=[],
            max_audio_length_ms=10_000,
        )
        return {"output": self.wav_to_base64(audio)}
