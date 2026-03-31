import logging
import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import base64
import io
import sys
import time

import nltk
import numpy as np
import scipy.io.wavfile as wav
import torch
from huggingface_hub import snapshot_download
from nltk.tokenize import sent_tokenize

nltk.download("punkt")


# Ensure data directory exists
os.makedirs("/app/data/Kokoro-82M", exist_ok=True)

# download model
snapshot_download(
    repo_id="hexgrad/Kokoro-82M",
    repo_type="model",
    revision="c97b7bbc3e60f447383c79b2f94fee861ff156ac",
    local_dir="/app/data/Kokoro-82M",
    ignore_patterns=["*.onnx", "kokoro-v0_19.pth", "demo/"],
    max_workers=8,
)
# add data_dir to the system path
sys.path.append("/app/data/Kokoro-82M")

from models import build_model

from kokoro import generate

logger = logging.getLogger(__name__)


class Model:
    def __init__(self, **kwargs):
        # Uncomment the following to get access
        # to various parts of the Truss config.

        self._data_dir = kwargs["data_dir"]
        self.model = None
        self.device = None
        self.default_voice = None
        self.voices = None
        # self._config = kwargs["config"]
        # self._secrets = kwargs["secrets"]
        return

    def load(self):
        logger.info("Starting setup...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Load model
        logger.info("Loading model...")
        model_path = "/app/data/Kokoro-82M/fp16/kokoro-v0_19-half.pth"
        logger.info(f"Model path: {model_path}")
        if not os.path.exists(model_path):
            logger.info(f"Error: Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")

        try:
            self.model = build_model(model_path, self.device)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.info(f"Error loading model: {str(e)}")
            raise

        # Load default voice
        logger.info("Loading default voice...")
        voice_path = "/app/data/Kokoro-82M/voices/af.pt"
        if not os.path.exists(voice_path):
            logger.info(f"Error: Voice file not found at {voice_path}")
            raise FileNotFoundError(f"Voice file not found at {voice_path}")

        try:
            self.default_voice = torch.load(voice_path).to(self.device)
            logger.info("Default voice loaded successfully")
        except Exception as e:
            logger.info(f"Error loading default voice: {str(e)}")
            raise

        # Dictionary of available voices
        self.voices = {
            "default": "af",
            "bella": "af_bella",
            "sarah": "af_sarah",
            "adam": "am_adam",
            "michael": "am_michael",
            "emma": "bf_emma",
            "isabella": "bf_isabella",
            "george": "bm_george",
            "lewis": "bm_lewis",
            "nicole": "af_nicole",
            "sky": "af_sky",
        }
        return

    def predict(self, model_input):
        # Run model inference here
        start = time.time()
        text = str(model_input.get("text", "Hi, I'm kokoro"))
        voice = str(model_input.get("voice", "af"))
        speed = float(model_input.get("speed", 1.0))
        logger.info(
            f"Text has {len(text)} characters. Using voice {voice} and speed {speed}."
        )
        if voice != "af":
            voicepack = torch.load(f"/app/data/Kokoro-82M/voices/{voice}.pt").to(
                self.device
            )
        else:
            voicepack = self.default_voice

        if len(text) >= 400:
            logger.info("Text is longer than 400 characters, splitting into sentences.")
            wavs = []

            def group_sentences(text, max_length=400):
                sentences = sent_tokenize(text)

                # Split long sentences
                while max([len(sent) for sent in sentences]) > max_length:
                    max_sent = max(sentences, key=len)
                    sentences_before = sentences[: sentences.index(max_sent)]
                    sentences_after = sentences[sentences.index(max_sent) + 1 :]

                    new_sentences = [
                        s.strip() + "." for s in max_sent.split(".") if s.strip()
                    ]
                    sentences = sentences_before + new_sentences + sentences_after

                return sentences

            sentences = group_sentences(text)
            logger.info(f"Processing {len(sentences)} chunks. Starting generation...")

            for sent in sentences:
                if sent.strip():
                    audio, _ = generate(
                        self.model, sent.strip(), voicepack, lang=voice[0], speed=speed
                    )
                    # Remove potential artifacts at the end
                    audio = audio[:-2000] if len(audio) > 2000 else audio
                    wavs.append(audio)

            # Concatenate all audio chunks
            audio = np.concatenate(wavs)
        else:
            logger.info("No splitting needed. Generating audio...")
            audio, _ = generate(self.model, text, voicepack, lang=voice[0], speed=speed)

        # Write audio to in-memory buffer
        buffer = io.BytesIO()
        wav.write(buffer, 24000, audio)
        wav_bytes = buffer.getvalue()
        duration_seconds = len(audio) / 24000
        logger.info(
            f"Generation took {time.time() - start} seconds to generate {duration_seconds:.2f} seconds of audio"
        )
        return {"base64": base64.b64encode(wav_bytes).decode("utf-8")}
