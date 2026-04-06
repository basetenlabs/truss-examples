import base64
import io
import logging
import os
import time

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import numpy as np
import scipy.io.wavfile as wav
import torch
from kokoro import KModel, KPipeline

logger = logging.getLogger(__name__)

VOICES = [
    # American English (female)
    "af_heart", "af_bella", "af_nicole", "af_aoede", "af_kore",
    "af_sarah", "af_nova", "af_sky", "af_alloy", "af_jessica", "af_river",
    # American English (male)
    "am_michael", "am_fenrir", "am_puck", "am_echo", "am_eric",
    "am_liam", "am_onyx", "am_santa", "am_adam",
    # British English (female)
    "bf_emma", "bf_isabella", "bf_alice", "bf_lily",
    # British English (male)
    "bm_george", "bm_fable", "bm_lewis", "bm_daniel",
]

DEFAULT_VOICE = "af_heart"


class Model:
    def __init__(self, **kwargs):
        self._data_dir = kwargs["data_dir"]
        self.device = None
        self.model = None
        self.pipelines = None

    def load(self):
        logger.info("Starting setup...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        logger.info("Loading KModel...")
        self.model = KModel().to(self.device).eval()
        logger.info("KModel loaded successfully")

        logger.info("Initializing pipelines...")
        # One pipeline per language code: 'a' = American English, 'b' = British English
        self.pipelines = {
            lang_code: KPipeline(lang_code=lang_code, model=self.model)
            for lang_code in "ab"
        }
        logger.info("Setup complete")

    def predict(self, model_input):
        start = time.time()
        text = str(model_input.get("text", "Hi, I'm Kokoro"))
        voice = str(model_input.get("voice", DEFAULT_VOICE))
        speed = float(model_input.get("speed", 1.0))

        if voice not in set(VOICES):
            raise ValueError(
                f"Unknown voice '{voice}'. Available voices: {VOICES}"
            )

        lang_code = voice[0]  # 'a' = American English, 'b' = British English
        logger.info(
            f"Generating {len(text)} characters with voice '{voice}' at speed {speed}."
        )

        pipeline = self.pipelines[lang_code]
        audio_chunks = []
        for _, _, audio in pipeline(text, voice=voice, speed=speed):
            audio_chunks.append(audio.numpy())

        audio = np.concatenate(audio_chunks)

        buffer = io.BytesIO()
        wav.write(buffer, 24000, audio)
        wav_bytes = buffer.getvalue()

        duration_seconds = len(audio) / 24000
        logger.info(
            f"Generated {duration_seconds:.2f}s of audio in {time.time() - start:.2f}s"
        )
        return {"base64": base64.b64encode(wav_bytes).decode("utf-8")}
