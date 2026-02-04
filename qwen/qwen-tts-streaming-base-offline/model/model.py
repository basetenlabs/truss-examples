"""
Qwen3-TTS Model using vLLM-Omni's Omni class for offline inference.

Supports:
- CustomVoice: Predefined speakers with optional style control
- VoiceDesign: Natural language voice style description  
- Base: Voice cloning from reference audio
"""

import base64
import io
import os
from typing import Any

import numpy as np
import soundfile as sf

MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs.get("data_dir")
        self._config = kwargs.get("config")
        self._secrets = kwargs.get("secrets")
        self._omni = None
        self._sampling_params = None
        self._task_type = None
        self._lazy_data_resolver = kwargs.get("lazy_data_resolver")

    def load(self) -> None:
        """Load the Qwen3-TTS model using vLLM-Omni."""
        from vllm import SamplingParams
        import vllm_omni
        from vllm_omni import Omni

        self._lazy_data_resolver.block_until_download_complete()

        model_path = MODEL_PATH
        # Determine task type from model path
        model_name = model_path.split("/")[-1]
        if "CustomVoice" in model_name:
            self._task_type = "CustomVoice"
        elif "VoiceDesign" in model_name:
            self._task_type = "VoiceDesign"
        else:
            self._task_type = "Base"

        print(f"Loading Qwen3-TTS model: {model_path}")
        print(f"Default task type: {self._task_type}")

        # Resolve stage config path from installed vllm_omni package
        stage_configs_path = os.path.join(
            os.path.dirname(vllm_omni.__file__),
            "model_executor", "stage_configs", "qwen3_tts.yaml",
        )

        # Initialize Omni engine
        self._omni = Omni(
            model=model_path,
            stage_configs_path=stage_configs_path,
            log_stats=False,
            stage_init_timeout=600,
        )

        # Default sampling params for TTS
        self._sampling_params = SamplingParams(
            temperature=0.9,
            top_p=1.0,
            top_k=50,
            max_tokens=2048,
            seed=None,
            detokenize=False,
            repetition_penalty=1.05,
        )

        print("Model loaded successfully")

    def predict(self, model_input: dict[str, Any]) -> dict[str, Any]:
        """
        Generate speech from text.

        Args:
            model_input: Dict with keys:
                - input/text: Text to synthesize (required)
                - task_type: "CustomVoice", "VoiceDesign", or "Base"
                - voice/speaker: Speaker name for CustomVoice
                - language: Language code (Auto, Chinese, English, etc.)
                - instructions/instruct: Style/emotion instructions
                - ref_audio: Reference audio URL for Base task
                - ref_text: Transcript of reference audio
                - x_vector_only_mode: Use speaker embedding only
                - response_format: Output format (wav, mp3, flac)

        Returns:
            Dict with audio_data (base64) and sample_rate
        """
        # Extract text
        text = model_input.get("input") or model_input.get("text")
        if not text:
            return {"error": "No input text provided"}

        # Get parameters
        task_type = model_input.get("task_type", self._task_type)
        language = model_input.get("language", "Auto")
        response_format = model_input.get("response_format", "wav")

        print(f"Generating speech: task_type={task_type}, text={text[:50]}...")

        try:
            # Build Omni input based on task type
            if task_type == "CustomVoice":
                omni_input = self._build_custom_voice_input(model_input, text, language)
            elif task_type == "VoiceDesign":
                omni_input = self._build_voice_design_input(model_input, text, language)
            elif task_type == "Base":
                omni_input = self._build_base_input(model_input, text, language)
            else:
                return {"error": f"Invalid task_type: {task_type}"}

            # Run generation
            sampling_params_list = [self._sampling_params]
            
            audio_tensor = None
            audio_samplerate = 24000

            for stage_outputs in self._omni.generate(omni_input, sampling_params_list):
                for output in stage_outputs.request_output:
                    audio_tensor = output.multimodal_output["audio"]
                    audio_samplerate = output.multimodal_output["sr"].item()

            if audio_tensor is None:
                return {"error": "No audio generated"}

            # Convert to numpy
            audio_numpy = audio_tensor.float().detach().cpu().numpy()
            if audio_numpy.ndim > 1:
                audio_numpy = audio_numpy.flatten()

            # Encode to requested format
            audio_bytes = self._encode_audio(audio_numpy, audio_samplerate, response_format)

            return {
                "audio_data": base64.b64encode(audio_bytes).decode("utf-8"),
                "sample_rate": audio_samplerate,
                "format": response_format,
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    def _build_custom_voice_input(
        self, model_input: dict, text: str, language: str
    ) -> dict:
        """Build input for CustomVoice task."""
        speaker = model_input.get("voice") or model_input.get("speaker", "Vivian")
        instruct = model_input.get("instructions") or model_input.get("instruct", "")
        max_tokens = model_input.get("max_new_tokens", 2048)

        prompt = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

        return {
            "prompt": prompt,
            "additional_information": {
                "task_type": ["CustomVoice"],
                "text": [text],
                "language": [language],
                "speaker": [speaker],
                "instruct": [instruct],
                "max_new_tokens": [max_tokens],
            },
        }

    def _build_voice_design_input(
        self, model_input: dict, text: str, language: str
    ) -> dict:
        """Build input for VoiceDesign task."""
        instruct = model_input.get("instructions") or model_input.get("instruct", "")
        max_tokens = model_input.get("max_new_tokens", 2048)

        if not instruct:
            raise ValueError("VoiceDesign task requires 'instructions'")

        prompt = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

        return {
            "prompt": prompt,
            "additional_information": {
                "task_type": ["VoiceDesign"],
                "text": [text],
                "language": [language],
                "instruct": [instruct],
                "max_new_tokens": [max_tokens],
                "non_streaming_mode": [True],
            },
        }

    def _build_base_input(
        self, model_input: dict, text: str, language: str
    ) -> dict:
        """Build input for Base (voice clone) task."""
        ref_audio = model_input.get("ref_audio")
        ref_text = model_input.get("ref_text", "")
        x_vector_only = model_input.get("x_vector_only_mode", False)
        max_tokens = model_input.get("max_new_tokens", 2048)

        prompt = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

        additional_info = {
            "task_type": ["Base"],
            "text": [text],
            "language": [language],
            "x_vector_only_mode": [x_vector_only],
            "max_new_tokens": [max_tokens],
        }

        # Add ref_audio if provided
        if ref_audio:
            additional_info["ref_audio"] = [ref_audio]
        
        # Add ref_text if provided and not x_vector_only mode
        if ref_text and not x_vector_only:
            additional_info["ref_text"] = [ref_text]

        return {
            "prompt": prompt,
            "additional_information": additional_info,
        }

    def _encode_audio(
        self, audio: np.ndarray, sample_rate: int, fmt: str = "wav"
    ) -> bytes:
        """Encode audio array to bytes in the requested format."""
        buffer = io.BytesIO()

        if fmt == "pcm":
            # Raw 16-bit PCM
            audio_int16 = (audio * 32767).astype(np.int16)
            buffer.write(audio_int16.tobytes())
        else:
            # Use soundfile for other formats
            sf.write(buffer, audio, sample_rate, format=fmt.upper())

        buffer.seek(0)
        return buffer.read()
