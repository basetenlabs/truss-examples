import base64
import io
import logging

import numpy as np
import scipy.io.wavfile as wav
import torch
from kokoro import KPipeline

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000
DEFAULT_VOICE = "af_heart"


class Model:
    def __init__(self, **kwargs):
        self._pipelines: dict[str, KPipeline] = {}
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self):
        logger.info(f"Loading Kokoro on {self._device}.")
        # American English by default; other languages load on demand in predict().
        self._pipelines["a"] = KPipeline(
            lang_code="a", repo_id="hexgrad/Kokoro-82M", device=self._device
        )
        # Warm the default voicepack so the first request doesn't pay the download cost.
        self._pipelines["a"].load_voice(DEFAULT_VOICE)
        logger.info("Kokoro loaded.")

    def _pipeline_for(self, lang_code: str) -> KPipeline:
        if lang_code not in self._pipelines:
            self._pipelines[lang_code] = KPipeline(
                lang_code=lang_code,
                repo_id="hexgrad/Kokoro-82M",
                device=self._device,
            )
        return self._pipelines[lang_code]

    def predict(self, model_input):
        text = str(model_input.get("text", "Hi, I'm Kokoro."))
        voice = str(model_input.get("voice", DEFAULT_VOICE))
        speed = float(model_input.get("speed", 1.0))

        # Voice prefix encodes language: a=American English, b=British English,
        # j=Japanese, z=Mandarin, e=Spanish, f=French, h=Hindi, i=Italian, p=Portuguese.
        pipeline = self._pipeline_for(voice[0])

        chunks = []
        for _, _, audio in pipeline(text, voice=voice, speed=speed):
            if audio is None:
                continue
            if hasattr(audio, "cpu"):
                audio = audio.cpu().numpy()
            chunks.append(audio)

        if not chunks:
            raise ValueError("No audio generated; check the input text and voice.")
        audio = np.concatenate(chunks)

        buffer = io.BytesIO()
        wav.write(buffer, SAMPLE_RATE, audio)
        return {"base64": base64.b64encode(buffer.getvalue()).decode("utf-8")}
