import base64
import io
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Tuple

import torch
from fam.llm.enhancers import get_enhancer
from fam.llm.sample import (
    InferenceConfig,
    Model,
    build_models,
    get_first_stage_path,
    get_second_stage_path,
    sample_utterance,
)
from fam.llm.utils import check_audio_file, get_default_dtype, get_default_use_kv_cache
from huggingface_hub import snapshot_download
from scipy.io import wavfile

HF_MODEL_ID = "metavoiceio/metavoice-1B-v0.1"


@dataclass
class DefaultInferenceParams:
    max_new_tokens: int = 864 * 2
    """Maximum number of new tokens to generate from the first stage model."""

    temperature: float = 1.0
    """Temperature for sampling applied to both models."""

    top_k: int = 200
    """Top k for sampling applied to both models."""

    seed: int = 1337
    """Random seed for sampling."""

    dtype: Literal["bfloat16", "float16", "float32", "tfloat32"] = get_default_dtype()
    """Data type to use for sampling."""

    enhancer: Optional[Literal["df"]] = "df"
    """Enhancer to use for post-processing."""

    use_kv_cache: Optional[
        Literal["flash_decoding", "vanilla"]
    ] = get_default_use_kv_cache()
    """Type of kv caching to use for inference: 1) [none] no kv caching, 2) [flash_decoding] use the
    flash decoding kernel, 3) [vanilla] use torch attention with hand implemented kv-cache."""


@dataclass
class TTSRequest:
    text: str
    guidance: Optional[Tuple[float, float]] = (3.0, 1.0)
    top_p: Optional[float] = 0.95
    speaker_ref_path: Optional[str] = None
    top_k: Optional[int] = None


@dataclass
class ModelState:
    """Data class for holding model state."""

    spkemb_model: Optional[torch.nn.Module]
    first_stage_model: Optional[Model]
    second_stage_model: Optional[Model]
    enhancer: Optional[object]


class Model:
    def __init__(self, **kwargs):
        self._secrets = kwargs["secrets"]
        self.config = DefaultInferenceParams()
        self.model_state = ModelState(None, None, None, None)
        self._data_dir = kwargs["data_dir"]

    def load(self):
        # kind of re-implement fam/llm/serving.py
        device = "cuda" if torch.cuda.is_available() else "cpu"

        common_config = dict(
            num_samples=1,
            seed=self.config.seed,
            device=device,
            dtype=self.config.dtype,
            compile=False,
            init_from="resume",
            output_dir=tempfile.mkdtemp(),
        )

        # Get models (from the cache, via config.yml)
        model_dir = snapshot_download(repo_id=HF_MODEL_ID)

        config1 = InferenceConfig(
            ckpt_path=get_first_stage_path(model_dir),
            **common_config,
        )

        config2 = InferenceConfig(
            ckpt_path=get_second_stage_path(model_dir),
            **common_config,
        )

        spkemb, llm_stg1, llm_stg2 = build_models(
            config1,
            config2,
            model_dir=model_dir,
            device=device,
            use_kv_cache=self.config.use_kv_cache,
        )

        self.model_state.spkemb_model = spkemb
        self.model_state.first_stage_model = llm_stg1
        self.model_state.second_stage_model = llm_stg2
        self.model_state.enhancer = get_enhancer(self.config.enhancer)

    def predict(self, model_input: Any) -> Any:
        audiodata = None  # optionally, extract reference audio file from model_input
        wav_out_path = None

        tts_req = TTSRequest(**model_input)

        # Support recommended Truss input format
        if model_input.get("prompt"):
            tts_req.text = model_input["prompt"]

        if tts_req.speaker_ref_path is None:
            tts_req.speaker_ref_path = os.path.join(self._data_dir, "bria.mp3")

        wav_path = tts_req.speaker_ref_path
        wav_out_path = sample_utterance(
            tts_req.text,
            wav_path,
            self.model_state.spkemb_model,
            self.model_state.first_stage_model,
            self.model_state.second_stage_model,
            enhancer=self.model_state.enhancer,
            first_stage_ckpt_path=None,
            second_stage_ckpt_path=None,
            guidance_scale=tts_req.guidance,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_k=tts_req.top_k,
            top_p=tts_req.top_p,
        )

        b64 = wav_to_b64(wav_out_path)
        Path(wav_out_path).unlink(missing_ok=True)
        return b64


def wav_to_b64(wav_in_path: str) -> str:
    SAMPLE_RATE, audio_array = wavfile.read(wav_in_path)

    with io.BytesIO() as byte_io:
        wavfile.write(byte_io, SAMPLE_RATE, audio_array)
        wav_bytes = byte_io.getvalue()

    audio_data = base64.b64encode(wav_bytes).decode("UTF-8")
    return audio_data
