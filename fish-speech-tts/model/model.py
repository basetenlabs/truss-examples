import base64
import io
import logging
import tempfile
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, Optional

import numpy as np
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
    """Text-to-speech model wrapper for Fish Speech (OpenAudio).

    This class provides an interface for generating speech from text,
    optionally using voice cloning with a reference audio prompt.
    """

    def __init__(self, **kwargs):
        self._llm_model = None
        self._decode_one_token = None
        self._codec_model = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._precision = torch.bfloat16
        self._checkpoint_path = Path("/app/checkpoints/openaudio-s1-mini")

    def load(self) -> None:
        """Loads the Fish Speech model."""
        import traceback
        import os
        
        try:
            logger.info(f"Python version: {os.sys.version}")
            logger.info(f"PyTorch version: {torch.__version__}")
            
            from huggingface_hub import snapshot_download
            logger.info("Successfully imported huggingface_hub")
            
            logger.info(f"Device: {self._device}")
            logger.info(f"Checkpoint path: {self._checkpoint_path}")
            logger.info(f"Checkpoint path exists: {self._checkpoint_path.exists()}")
            
            logger.info("Downloading Fish Speech model weights...")
            if not self._checkpoint_path.exists():
                logger.info("Creating checkpoint directory")
                os.makedirs(self._checkpoint_path, exist_ok=True)
                
                try:
                    snapshot_download(
                        repo_id="fishaudio/openaudio-s1-mini",
                        local_dir=str(self._checkpoint_path),
                    )
                    logger.info("Download complete")
                except Exception as e:
                    logger.error(f"Error downloading model weights: {e}")
                    logger.error(traceback.format_exc())
            
            try:
                logger.info("Importing fish_speech modules...")
                import fish_speech
                logger.info(f"Fish Speech version: {getattr(fish_speech, '__version__', 'unknown')}")
                
                from fish_speech.models.text2semantic.inference import init_model
                from fish_speech.models.dac.inference import load_model as load_codec
                logger.info("Successfully imported fish_speech modules")
                
                logger.info("Loading Fish Speech LLM model...")
                self._llm_model, self._decode_one_token = init_model(
                    checkpoint_path=str(self._checkpoint_path),
                    device=self._device,
                    precision=self._precision,
                    compile=False,
                )
                logger.info("LLM model loaded")
                
                with torch.device(self._device):
                    logger.info(f"Setting up caches with max_seq_len={self._llm_model.config.max_seq_len}")
                    self._llm_model.setup_caches(
                        max_batch_size=1,
                        max_seq_len=self._llm_model.config.max_seq_len,
                        dtype=self._precision,
                    )
                    logger.info("Caches set up")
                
                logger.info("Loading Fish Speech codec model...")
                self._codec_model = load_codec(
                    config_name="modded_dac_vq",
                    checkpoint_path=str(self._checkpoint_path / "codec.pth"),
                    device=self._device,
                )
                logger.info("Codec model loaded")
                
                logger.info("Fish Speech models loaded successfully")
            except Exception as e:
                logger.error(f"Error loading models: {e}")
                logger.error(traceback.format_exc())
                raise
        except Exception as e:
            logger.error(f"Fatal error in load(): {e}")
            logger.error(traceback.format_exc())
            raise

    def _encode_reference_audio(self, audio_path: Path) -> torch.Tensor:
        """Encode reference audio to VQ tokens."""
        audio, sr = torchaudio.load(str(audio_path))
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        audio = torchaudio.functional.resample(audio, sr, self._codec_model.sample_rate)
        
        audios = audio[None].to(self._device)
        audio_lengths = torch.tensor([audios.shape[2]], device=self._device, dtype=torch.long)
        
        indices, _ = self._codec_model.encode(audios, audio_lengths)
        if indices.ndim == 3:
            indices = indices[0]
        
        return indices

    def _decode_codes_to_audio(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode semantic codes to audio."""
        indices_lens = torch.tensor([codes.shape[1]], device=self._device, dtype=torch.long)
        fake_audios, _ = self._codec_model.decode(codes, indices_lens)
        return fake_audios[0, 0]

    def predict(self, model_input: Dict[str, str]) -> Dict[str, str]:
        """Generates speech from text with optional voice cloning.

        Args:
            model_input: Dictionary containing:
                - text: The text to convert to speech
                - voice: Optional base64 encoded audio for voice cloning
                - voice_text: Optional transcript of the voice audio (improves cloning)

        Returns:
            Dict containing the generated audio as a base64 encoded string
        """
        from fish_speech.models.text2semantic.inference import generate_long

        text = model_input["text"]
        voice_b64 = model_input.get("voice")
        voice_text = model_input.get("voice_text")

        prompt_tokens = None
        prompt_text = None
        
        if voice_b64:
            logger.info("Using voice audio for cloning...")
            with temp_audio_file(voice_b64) as voice_path:
                prompt_tokens = [self._encode_reference_audio(voice_path)]
                prompt_text = [voice_text] if voice_text else [""]

        all_codes = []
        generator = generate_long(
            model=self._llm_model,
            device=self._device,
            decode_one_token=self._decode_one_token,
            text=text,
            num_samples=1,
            max_new_tokens=0,
            top_p=0.8,
            repetition_penalty=1.1,
            temperature=0.8,
            compile=False,
            iterative_prompt=True,
            chunk_length=300,
            prompt_text=prompt_text,
            prompt_tokens=prompt_tokens,
        )

        for response in generator:
            if response.action == "sample":
                all_codes.append(response.codes)
            elif response.action == "next":
                break

        if not all_codes:
            raise ValueError("No audio generated")

        codes = torch.cat(all_codes, dim=1)
        audio = self._decode_codes_to_audio(codes)

        buffer = io.BytesIO()
        torchaudio.save(buffer, audio.cpu().unsqueeze(0), self._codec_model.sample_rate, format="wav")
        buffer.seek(0)
        wav_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        return {"audio": wav_base64}
