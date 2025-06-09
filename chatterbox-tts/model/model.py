import base64
import io
import logging
from pathlib import Path
import tempfile
from contextlib import contextmanager
from typing import Dict, Optional, Union

import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@contextmanager
def temp_audio_file(audio_b64: str) -> Path:
    """Creates a temporary audio file from a base64 encoded string.
    
    Args:
        audio_b64: Base64 encoded audio data
        
    Yields:
        Path: Path to the temporary audio file
        
    Raises:
        ValueError: If the base64 string is invalid
    """
    try:
        audio_bytes = base64.b64decode(audio_b64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_file_path = Path(temp_file.name)
        
        yield temp_file_path
    finally:
        try:
            temp_file_path.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to delete temporary file {temp_file_path}: {e}")

class Model:
    """Text-to-speech model wrapper for ChatterboxTTS.
    
    This class provides an interface for generating speech from text,
    optionally using voice cloning with an audio prompt.
    """
    def __init__(self, **kwargs):
        self._model = None

    def load(self) -> None:
        """Loads the ChatterboxTTS model on CUDA device."""
        self._model = ChatterboxTTS.from_pretrained(device="cuda")

    def predict(self, model_input: Dict[str, str]) -> Dict[str, str]:
        """Generates speech from text with optional voice cloning.
        
        Args:
            model_input: Dictionary containing:
                - text: The text to convert to speech
                - audio_prompt: Optional base64 encoded audio for voice cloning
                
        Returns:
            Dict containing the generated audio as a base64 encoded string
            
        Raises:
            ValueError: If the model is not loaded or input is invalid
        """
            
        text = model_input["text"]
        audio_prompt_b64 = model_input.get("audio_prompt")
        
        if audio_prompt_b64:
            logger.info("Using audio prompt for voice cloning...")
            with temp_audio_file(audio_prompt_b64) as audio_path:
                wav = self._model.generate(text, audio_prompt_path=audio_path)
        else:
            wav = self._model.generate(text)
        
        buffer = io.BytesIO()
        ta.save(buffer, wav, self._model.sr, format="wav")
        buffer.seek(0)
        wav_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        return {"audio": wav_base64}
