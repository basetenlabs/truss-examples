import base64
import tempfile
import torch
import io
import soundfile as sf

from audiocraft.data.audio import audio_write
from audiocraft.models import MusicGen

TARGET_SAMPLE_RATE = 32000
TARGET_AUDIO_CHANNELS = 1

class Model:
    def load(self):
        self.model = MusicGen.get_pretrained("melody", device="cuda")

    def _decode_base64_to_wav(self, base64_string):
        """Decode a base64 string to a waveform."""
        byte_data = base64.b64decode(base64_string)
        with io.BytesIO(byte_data) as f:
            waveform, sample_rate = sf.read(f)
        return waveform, sample_rate

    def _convert_audio(self, melody, sr):
        """Convert audio to the target sample rate and channel."""
        # Convert to torch tensor and transpose
        melody = torch.from_numpy(melody).to(self.model.device).float().t()
        if melody.dim() == 1:
            melody = melody[None]

        # Convert to mono if stereo
        if melody.shape[1] == 2:
            melody = torch.mean(melody, dim=1, keepdim=True)
        return melody

    def predict(self, request):
        try:
            prompts = request.pop("prompts")
            duration = request.pop("duration") if "duration" in request else 8
            self.model.set_generation_params(duration=duration)

            melody_base64 = request.pop("melody", None)

            if melody_base64:
                melody, sample_rate = self._decode_base64_to_wav(melody_base64)
                processed_melody = self._convert_audio(melody, sample_rate)
                wav = self.model.generate_with_chroma(
                    descriptions=prompts,
                    melody_wavs=[processed_melody for _ in prompts],
                    melody_sample_rate=TARGET_SAMPLE_RATE
                )
            else:
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
