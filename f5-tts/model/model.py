import base64
import io
import logging
import tempfile
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, Optional

import torch
import torchaudio
import soundfile as sf

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@contextmanager
def temp_audio_file(audio_b64: str, suffix: str = ".wav"):
    """Creates a temporary audio file from a base64 encoded string."""
    temp_file_path = None
    try:
        audio_bytes = base64.b64decode(audio_b64)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_file_path = Path(temp_file.name)
        yield temp_file_path
    finally:
        if temp_file_path:
            try:
                temp_file_path.unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_file_path}: {e}")


class Model:
    """F5-TTS Text-to-Speech model with zero-shot voice cloning.

    F5-TTS uses Flow Matching for high-quality voice cloning from a short
    audio sample. It can generate natural-sounding speech that matches
    the voice characteristics of the reference audio.
    """

    def __init__(self, **kwargs):
        self._model = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self) -> None:
        """Loads the F5-TTS model."""
        from f5_tts.api import F5TTS
        
        logger.info(f"Device: {self._device}")
        logger.info("Loading F5-TTS model...")
        
        self._model = F5TTS(device=self._device)
        
        logger.info("F5-TTS model loaded successfully")

    def predict(self, model_input: Dict[str, str]) -> Dict[str, str]:
        """Generates speech from text with voice cloning.

        Args:
            model_input: Dictionary containing:
                - text: The text to convert to speech
                - voice: Base64 encoded audio for voice cloning
                - ref_text: Optional transcript of the reference audio (improves quality)

        Returns:
            Dict containing the generated audio as a base64 encoded string
        """
        text = model_input["text"]
        voice_b64 = model_input.get("voice")
        ref_text = model_input.get("ref_text", "")
        
        if not voice_b64:
            raise ValueError("voice (base64 encoded audio) is required for voice cloning")
        
        with temp_audio_file(voice_b64) as reference_audio_path:
            logger.info("Generating speech with F5-TTS...")
            
            result = self._model.infer(
                ref_file=str(reference_audio_path),
                ref_text=ref_text,
                gen_text=text,
            )
            # F5-TTS returns (audio, sample_rate, additional_info)
            audio = result[0]
            sample_rate = 24000  # F5-TTS uses fixed 24kHz
            
            buffer = io.BytesIO()
            sf.write(buffer, audio, sample_rate, format="wav")
            buffer.seek(0)
            wav_base64 = base64.b64encode(buffer.read()).decode("utf-8")
            
            return {"audio": wav_base64}
