import base64
import io
import logging
import os

import soundfile as sf
import torch
from fastapi import Response
from qwen_tts import Qwen3TTSModel

logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

DEFAULT_GEN_KWARGS = dict(
    do_sample=True,
    top_k=50,
    top_p=1.0,
    temperature=0.9,
    repetition_penalty=1.05,
    subtalker_dosample=True,
    subtalker_top_k=50,
    subtalker_top_p=1.0,
    subtalker_temperature=0.9,
)

GEN_KWARG_KEYS = (
    "top_k",
    "top_p",
    "temperature",
    "repetition_penalty",
    "subtalker_top_k",
    "subtalker_top_p",
    "subtalker_temperature",
)


class Model:
    def __init__(self, **kwargs):
        self._data_dir = kwargs["data_dir"]
        self._secrets = kwargs["secrets"]
        self.tts = None

    def load(self):
        os.environ["HF_TOKEN"] = self._secrets["hf_access_token"]
        self.tts = Qwen3TTSModel.from_pretrained(
            MODEL_ID,
            device_map="cuda:0",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        logger.info("Qwen3-TTS Base model loaded")

    def _build_gen_kwargs(self, model_input: dict) -> dict:
        kwargs = dict(DEFAULT_GEN_KWARGS)
        kwargs["max_new_tokens"] = model_input.get("max_new_tokens", 2048)
        for key in GEN_KWARG_KEYS:
            if key in model_input:
                kwargs[key] = model_input[key]
        return kwargs

    def _wav_bytes(self, audio_data, sr: int, fmt: str = "WAV") -> bytes:
        buf = io.BytesIO()
        sf.write(buf, audio_data, sr, format=fmt)
        buf.seek(0)
        return buf.read()

    async def predict(self, model_input: dict):
        text = model_input.get("input", model_input.get("text", ""))
        is_batch = isinstance(text, list)

        language = model_input.get("language", "Auto")
        ref_audio = model_input.get("ref_audio")
        ref_text = model_input.get("ref_text")
        x_vector_only_mode = model_input.get("x_vector_only_mode", False)

        response_format = model_input.get("response_format", "wav").upper()
        if response_format not in ("WAV", "FLAC"):
            response_format = "WAV"

        if not ref_audio:
            return Response(
                content='{"error": "ref_audio is required for voice cloning"}',
                media_type="application/json",
                status_code=400,
            )

        if not x_vector_only_mode and not ref_text:
            return Response(
                content='{"error": "ref_text is required for ICL voice cloning (or use x_vector_only_mode)"}',
                media_type="application/json",
                status_code=400,
            )

        gen_kwargs = self._build_gen_kwargs(model_input)

        wavs, sr = self.tts.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only_mode,
            **gen_kwargs,
        )

        if not is_batch:
            audio_bytes = self._wav_bytes(wavs[0], sr, fmt=response_format)
            return Response(content=audio_bytes, media_type="audio/wav")

        encoded = []
        for w in wavs:
            b64 = base64.b64encode(self._wav_bytes(w, sr, fmt=response_format)).decode()
            encoded.append(b64)

        return {"audio": encoded, "sample_rate": sr, "count": len(encoded)}
