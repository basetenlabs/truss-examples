from transformers import AutoConfig, AutoModel, AutoProcessor
from qwen_tts.core.models import (
    Qwen3TTSConfig,
    Qwen3TTSForConditionalGeneration,
    Qwen3TTSProcessor
)
import os
import subprocess
import sys

# Resolve stage config path from installed vllm_omni package
import vllm_omni

# Register model type
AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)
print("Registered qwen3_tts model type with transformers")


_stage_configs_path = os.path.join(
    os.path.dirname(vllm_omni.__file__),
    "model_executor", "stage_configs", "qwen3_tts.yaml"
)
if not os.path.isfile(_stage_configs_path):
    raise FileNotFoundError(f"Stage config not found: {_stage_configs_path}")

# Launch vllm-omni server
model_path = f"/app/model_cache/{os.environ['MODEL_TYPE']}"
sys.exit(subprocess.call([
    "vllm-omni", "serve", model_path,
    "--stage-configs-path", _stage_configs_path,
    "--host", "0.0.0.0",
    "--port", "8000",
    "--gpu-memory-utilization", "0.9",
    "--trust-remote-code",
    "--omni"
]))
