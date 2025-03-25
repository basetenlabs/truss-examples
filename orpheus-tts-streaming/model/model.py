import logging
import os
import torch
import struct
from fastapi.responses import StreamingResponse

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
# os.environ["VLLM_USE_V1"] = "1"
# os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"

from orpheus_tts.engine_class import OrpheusModel

logger = logging.getLogger(__name__)


class Model:
    def __init__(self, **kwargs):
        # Uncomment the following to get access
        # to various parts of the Truss config.

        self._data_dir = kwargs["data_dir"]
        self.model = None
        self._secrets = kwargs["secrets"]
        os.environ["HF_TOKEN"] = self._secrets["hf_access_token"]

    def load(self):
        # default dtype is torch.bfloat16
        # https://github.com/canopyai/Orpheus-Speech-PyPi/blob/main/orpheus_tts/engine_class.py#L10
        self.model = OrpheusModel(model_name = "canopylabs/orpheus-tts-0.1-finetune-prod", dtype=torch.float16)

    def create_wav_header(self, sample_rate=24000, bits_per_sample=16, channels=1):
        byte_rate = sample_rate * channels * bits_per_sample // 8
        block_align = channels * bits_per_sample // 8

        data_size = 0

        header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF',
            36 + data_size,
            b'WAVE',
            b'fmt ',
            16,
            1,
            channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            b'data',
            data_size
        )
        return header

    def predict(self, model_input):
        # Run model inference here
        text = str(model_input.get("text", "Hi, I'm Orhpeus model"))
        voice = str(model_input.get("voice", "tara"))
        request_id = str(model_input.get("request_id", "req-001"))
        repetition_penalty = model_input.get("repetition_penalty", 1.1)
        max_tokens = int(model_input.get("max_tokens", 10000))
        temperature = model_input.get("temperature", 0.4)
        top_p = model_input.get("top_p", 0.9)

        logger.info(
            f"Generating audio from processed text ({len(text)} chars, voice {voice}): {text}")

        def generate_audio_stream():
            yield self.create_wav_header()

            audio_generator = self.model.generate_speech(
                prompt=text,
                voice=voice,
                request_id=request_id,
                repetition_penalty=repetition_penalty,
                stop_token_ids=[128258],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            for chunk in audio_generator:
                yield chunk

        return StreamingResponse(
            generate_audio_stream(),
            media_type="audio/wav"
        )
