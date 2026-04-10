import base64
import io
import logging
import tempfile
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, Optional

import torch
import torchaudio

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
    """OpenVoice V2 Text-to-Speech model with voice cloning.

    This model provides high-quality voice cloning from a short audio sample,
    supporting multiple languages including English, Spanish, French, Chinese,
    Japanese, and Korean.
    """

    def __init__(self, **kwargs):
        self._tone_color_converter = None
        self._tts_model = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._checkpoint_path = Path("/app/checkpoints_v2")
        self._output_dir = Path("/tmp/openvoice_output")

    def load(self) -> None:
        """Loads the OpenVoice V2 model and MeloTTS."""
        import os
        os.makedirs(self._output_dir, exist_ok=True)
        
        logger.info(f"Device: {self._device}")
        logger.info(f"Checkpoint path: {self._checkpoint_path}")
        
        from openvoice import se_extractor
        from openvoice.api import ToneColorConverter
        from melo.api import TTS
        
        logger.info("Loading OpenVoice ToneColorConverter...")
        ckpt_converter = str(self._checkpoint_path / "converter")
        self._tone_color_converter = ToneColorConverter(
            f"{ckpt_converter}/config.json", 
            device=self._device
        )
        self._tone_color_converter.load_ckpt(f"{ckpt_converter}/checkpoint.pth")
        
        logger.info("Loading MeloTTS model...")
        self._tts_model = TTS(language="EN", device=self._device)
        
        self._se_extractor = se_extractor
        
        logger.info("OpenVoice V2 models loaded successfully")

    def predict(self, model_input: Dict[str, str]) -> Dict[str, str]:
        """Generates speech from text with voice cloning.

        Args:
            model_input: Dictionary containing:
                - text: The text to convert to speech
                - voice: Base64 encoded audio for voice cloning
                - language: Optional language code (EN, ES, FR, ZH, JP, KR). Default: EN

        Returns:
            Dict containing the generated audio as a base64 encoded string
        """
        import os
        import uuid
        
        text = model_input["text"]
        voice_b64 = model_input.get("voice")
        language = model_input.get("language", "EN").upper()
        speed = model_input.get("speed", 1.0)
        
        if not voice_b64:
            raise ValueError("voice (base64 encoded audio) is required for voice cloning")
        
        session_id = str(uuid.uuid4())[:8]
        
        with temp_audio_file(voice_b64) as reference_audio_path:
            logger.info(f"Extracting speaker embedding from reference audio...")
            
            target_se, _ = self._se_extractor.get_se(
                str(reference_audio_path), 
                self._tone_color_converter, 
                vad=False
            )
            
            src_path = self._output_dir / f"tmp_{session_id}.wav"
            
            logger.info(f"Generating base TTS audio with MeloTTS (language: {language})...")
            
            speaker_ids = self._tts_model.hps.data.spk2id
            speaker_key = list(speaker_ids.keys())[0]
            speaker_id = speaker_ids[speaker_key]
            
            self._tts_model.tts_to_file(
                text, 
                speaker_id, 
                str(src_path), 
                speed=speed
            )
            
            source_se = torch.load(
                str(self._checkpoint_path / "base_speakers" / "ses" / f"{language.lower()}.pth"),
                map_location=self._device
            )
            
            output_path = self._output_dir / f"output_{session_id}.wav"
            
            logger.info("Applying voice conversion...")
            self._tone_color_converter.convert(
                audio_src_path=str(src_path),
                src_se=source_se,
                tgt_se=target_se,
                output_path=str(output_path),
            )
            
            audio, sr = torchaudio.load(str(output_path))
            
            buffer = io.BytesIO()
            torchaudio.save(buffer, audio, sr, format="wav")
            buffer.seek(0)
            wav_base64 = base64.b64encode(buffer.read()).decode("utf-8")
            
            try:
                src_path.unlink(missing_ok=True)
                output_path.unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp files: {e}")
            
            return {"audio": wav_base64}
