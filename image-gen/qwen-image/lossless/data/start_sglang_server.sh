#!/bin/bash
# start_sglang_server.sh - Start the SGLang server
set -e

log() {
    echo "[SERVER $(date '+%H:%M:%S')] $1"
}

log "=========================================="
log "Starting Qwen-Image-2512 Lossless"
log "=========================================="

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="/workspace/flash-attention:$PYTHONPATH"
export HF_TOKEN=$(cat /secrets/hf_access_token)
export SGLANG_ENABLE_FP8_QUANTIZATION=0
export SGLANG_DIFFUSION_USE_CUTE_DSL_FLASH_ATTN=1
export SGLANG_DIFFUSION_ENABLE_CUDA_GRAPH_CAPTURE=0

unset B10_CPU_MEMORY_SAVING
/workspace/venv/bin/sglang serve \
    --model-path /app/model_cache/qwen-image-2512 \
    --dit-cpu-offload false \
    --text-encoder-cpu-offload false \
    --image-encoder-cpu-offload false \
    --vae-cpu-offload false \
    --num-gpus 1 \
    --port 8000 \
    --host 0.0.0.0
