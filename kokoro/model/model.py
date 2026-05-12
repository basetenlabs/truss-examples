import base64
import io
import logging
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wav
import torch
from kokoro import KModel, KPipeline

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000
DEFAULT_VOICE = "af_heart"
REPO_ID = "hexgrad/Kokoro-82M"
WEIGHTS_DIR = Path("/weights/kokoro")


class Model:
    def __init__(self, **kwargs):
        self._pipelines: dict[str, KPipeline] = {}
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._km: KModel | None = None
        self._voicepacks: dict[str, torch.FloatTensor] = {}

    def load(self):
        logger.info(f"Loading Kokoro from {WEIGHTS_DIR} on {self._device}.")
        self._km = (
            KModel(
                repo_id=REPO_ID,
                config=str(WEIGHTS_DIR / "config.json"),
                model=str(WEIGHTS_DIR / "kokoro-v1_0.pth"),
            )
            .to(self._device)
            .eval()
        )
        # Pre-read every voicepack mounted by BDN so requests never reach HF.
        for voice_file in (WEIGHTS_DIR / "voices").glob("*.pt"):
            self._voicepacks[voice_file.stem] = torch.load(
                str(voice_file), weights_only=True
            )
        # American English by default; other languages load on demand.
        self._pipelines["a"] = self._make_pipeline("a")
        logger.info(f"Kokoro loaded with {len(self._voicepacks)} voicepacks.")

    def _make_pipeline(self, lang_code: str) -> KPipeline:
        pipeline = KPipeline(lang_code=lang_code, repo_id=REPO_ID, model=self._km)
        pipeline.voices.update(self._voicepacks)
        return pipeline

    def _pipeline_for(self, lang_code: str) -> KPipeline:
        if lang_code not in self._pipelines:
            self._pipelines[lang_code] = self._make_pipeline(lang_code)
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
