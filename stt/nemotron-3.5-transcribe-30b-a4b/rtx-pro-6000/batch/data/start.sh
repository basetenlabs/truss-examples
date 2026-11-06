#!/usr/bin/env bash
# Serve Nemotron-3.5-Transcribe-30B-A4B with NVIDIA's model-card vLLM flags.
# BASETEN_MODEL_PATH is the base checkpoint mount, or the merged checkpoint a
# Baseten Training deployment mounts instead.
set -euo pipefail
: "${BASETEN_MODEL_PATH:?BASETEN_MODEL_PATH must name the checkpoint directory}"

VENV=/opt/nemo-speech/venv
# FlashInfer (ReplaySSM module) and Triton JIT-compile at startup with the
# image's CUDA toolkit; FlashInfer's fused-MoE and sampling kernels come
# prebuilt from flashinfer-jit-cache (config.yaml).
export CUDA_HOME=/usr/local/cuda
export PATH="${VENV}/bin:${CUDA_HOME}/bin:${PATH}"

# Cold engine init takes ~400 s (749 s before the prebuilt kernels). The BDN startup cache
# (ORG_ENABLE_BDN_STARTUP_CACHE) sets the JIT cache paths (VLLM_CACHE_ROOT,
# TORCHINDUCTOR_CACHE_DIR, TRITON_CACHE_DIR, FLASHINFER_WORKSPACE_BASE, ...)
# and restores them across replicas and restarts, keyed by GPU type.

# config.json names pretrained_llm as the relative path "llm_backbone".
cd "${BASETEN_MODEL_PATH}"

# Card flags for vLLM 0.32.0: ReplaySSM with the FlashInfer Mamba backend and
# prefix caching off. Deviation from the card: thinking off by default (clients
# that omit chat_template_kwargs get a transcript). NeMo Speech main registers
# the current architecture name, so no --hf-overrides. MoE backend: vLLM's
# auto selection.
exec "${VENV}/bin/vllm" serve "${BASETEN_MODEL_PATH}" \
    --tokenizer "${BASETEN_MODEL_PATH}" \
    --served-model-name nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.94 \
    --max-model-len 65536 \
    --max-num-seqs "${NEMOTRON_MAX_NUM_SEQS:-384}" \
    --max-num-batched-tokens "${NEMOTRON_MAX_NUM_BATCHED_TOKENS:-16384}" \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --use-replayssm \
    --mamba-backend flashinfer \
    --mamba-cache-mode none \
    --mm-processor-cache-gb 0 \
    --no-async-scheduling \
    --disable-uvicorn-access-log \
    --limit-mm-per-prompt '{"audio":{"count":1,"length":120000000}}' \
    --speculative-config '{"method":"mtp","num_speculative_tokens":8}' \
    --default-chat-template-kwargs '{"enable_thinking":false}' \
    --reasoning-parser nemotron_v3 \
    --trust-request-mm-kwargs \
    --middleware nemo.collections.speechlm2.vllm.salm.ctc_serving.ctc_timestamp_middleware
