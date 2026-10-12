"""Launch `vllm-omni` with stored ICL voice profiles allowed on CustomVoice checkpoints.

vLLM-Omni 0.30.0 routes every stored voice through the Base / in-context-cloning path, then
rejects it on a CustomVoice checkpoint: "Qwen3-TTS CustomVoice checkpoint does not support
task_type='Base'". The gate exists because Base requests that run the speaker encoder crash on
CustomVoice checkpoints (vllm-omni#6113). A precomputed `custom_voice_dir` profile already
holds its speaker embedding and reference codes, so the talker builds its Base/ICL prompt (with
the `instructions` prefix) without the encoder, as earlier releases allowed.

This skips only the checkpoint-variant check, and only for precomputed profiles that the
talker can load: it checks profile embeddings against `speaker_encoder_config.enc_dim`, which
defaults to 1024 when a CustomVoice config.json omits it, while the API server checks them
against the talker hidden size (2048 for 1.7B). On a mismatch the talker skips the profile and
a request for it kills EngineCore, so such checkpoints keep the stock 400 instead. Declare
`"speaker_encoder_config": {"enc_dim": 2048}` in the checkpoint's config.json to enable them.
Uploaded voices, `ref_audio` requests and built-in speakers validate exactly as before.

Usage (same arguments as `vllm-omni`): python serve_patched.py serve <model> ...
"""

import sys

from vllm_omni.entrypoints.cli.main import main
from vllm_omni.entrypoints.openai.tts_adapters import qwen3_tts

_Adapter = qwen3_tts.Qwen3TTSAdapter
_validate = _Adapter.validate
_get_model_variant = _Adapter._get_model_variant


def _patched_get_model_variant(self):
    if getattr(self, "_skip_variant_check", False):
        return None
    return _get_model_variant(self)


def _talker_loads_profiles(self) -> bool:
    hf_config = self.ctx.server.engine_client.model_config.hf_config
    enc_dim = getattr(getattr(hf_config, "speaker_encoder_config", None), "enc_dim", None)
    return enc_dim == hf_config.talker_config.hidden_size


def _patched_validate(self, request):
    voice = (request.voice or "").lower()
    server = self.ctx.server
    # Precomputed profiles carry their speaker embedding (and ICL codes), so no speaker encoder
    # runs. Uploaded audio voices still need the Base encoder and stay rejected.
    precomputed = (
        bool(voice)
        and voice in self.capabilities.precomputed_speakers
        and _talker_loads_profiles(self)
    )
    # validate() is synchronous, so no other request can observe the flag.
    self._skip_variant_check = precomputed
    try:
        return _validate(self, request)
    finally:
        self._skip_variant_check = False


_Adapter._get_model_variant = _patched_get_model_variant
_Adapter.validate = _patched_validate

if __name__ == "__main__":
    sys.argv[0] = "vllm-omni"
    sys.exit(main())
