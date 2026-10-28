#!/usr/bin/env bash
# Serve Nemotron-3.5-Transcribe-30B-A4B with NVIDIA's model-card vLLM flags.
# BASETEN_MODEL_PATH is the base checkpoint mount, or the merged checkpoint a
# Baseten Training deployment mounts instead.
set -euo pipefail
: "${BASETEN_MODEL_PATH:?BASETEN_MODEL_PATH must name the checkpoint directory}"

VENV=/opt/nemo-speech/venv
# FlashInfer and Triton JIT-compile at startup with the image's CUDA toolkit.
export CUDA_HOME=/usr/local/cuda
export PATH="${VENV}/bin:${CUDA_HOME}/bin:${PATH}"

# The cold JIT warmup takes ~20 min; b10cache (/cache/org) keeps it across
# replicas and restarts when enabled, else fall back to local disk.
CACHE_BASE=/cache/org/nemotron-3.5-transcribe-30b-a4b
mkdir -p "${CACHE_BASE}" 2>/dev/null && [[ -w "${CACHE_BASE}" ]] || CACHE_BASE=/root/.cache/nemotron-3.5-transcribe-30b-a4b
export FLASHINFER_WORKSPACE_BASE="${CACHE_BASE}"
export VLLM_CACHE_ROOT="${CACHE_BASE}/vllm"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_BASE}/inductor"
export TRITON_CACHE_DIR="${CACHE_BASE}/triton"
mkdir -p "${VLLM_CACHE_ROOT}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

# config.json names pretrained_llm as the relative path "llm_backbone".
cd "${BASETEN_MODEL_PATH}"

# The card's --mamba-cache-mode all resolves to none: vLLM 0.28.0 keeps prefix
# caching off for this model. Deviations from the card: thinking off by default
# (clients that omit chat_template_kwargs get a transcript) and the CTC
# timestamp middleware. The public config.json renamed the architecture; the
# pinned Speech commit registers only the old name.
exec "${VENV}/bin/vllm" serve "${BASETEN_MODEL_PATH}" \
    --tokenizer "${BASETEN_MODEL_PATH}" \
    --served-model-name nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --hf-overrides '{"architectures":["NeMoSpeechLMForConditionalGeneration"]}' \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.94 \
    --max-model-len 65536 \
    --max-num-seqs 192 \
    --max-num-batched-tokens 4096 \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --mamba-cache-mode none \
    --mm-processor-cache-gb 0 \
    --no-async-scheduling \
    --disable-uvicorn-access-log \
    --limit-mm-per-prompt '{"audio":{"count":1,"length":120000000}}' \
    --speculative-config '{"method":"mtp","num_speculative_tokens":8}' \
    --default-chat-template-kwargs '{"enable_thinking":false}' \
    --reasoning-parser nemotron_v3 \
    --middleware nemo.collections.speechlm2.vllm.salm.ctc_serving.ctc_timestamp_middleware
