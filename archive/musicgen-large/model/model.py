import base64
import tempfile

from audiocraft.data.audio import audio_write
from audiocraft.models import MusicGen

MODEL_SIZE = "large"


class Model:
    def load(self):
        self.model = MusicGen.get_pretrained(MODEL_SIZE, device="cuda")

    def predict(self, request):
        try:
            prompts = request.pop("prompts")
            duration = request.pop("duration") if "duration" in request else 8
            self.model.set_generation_params(duration=duration)
            wav = self.model.generate(prompts)
            output_files = []
            for idx, one_wav in enumerate(wav):
                with tempfile.NamedTemporaryFile() as tmpfile:
                    audio_write(
                        tmpfile.name,
                        one_wav.cpu(),
                        self.model.sample_rate,
                        strategy="loudness",
                    )
                    with open(tmpfile.name + ".wav", "rb") as f:
                        output_files.append(base64.b64encode(f.read()).decode("utf-8"))

            return {"data": output_files}
        except Exception as exc:
            return {
                "status": "error",
                "data": None,
                "message": str(exc),
                "traceback": str(exc.__traceback__),
            }
