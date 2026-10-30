#!/usr/bin/env bash
# The engine sidecar for the unified streaming preset. nemotron_stt's build() launches this and
# waits for its /health before returning the backend, so baseten-stt (the image's Python 3.12)
# never imports NeMo or vLLM: those live in their own Python 3.13 venv at /opt/nemo-speech,
# built as the batch preset's (stt/nemotron-3.5-transcribe-30b-a4b/rtx-pro-6000/batch). Same
# weights and NVIDIA model-card flags as that preset's data/start.sh, minus the CTC timestamp
# middleware (turns are plain text); it binds to localhost only, since nothing outside the
# container talks to it.
set -euo pipefail
: "${BASETEN_MODEL_PATH:?BASETEN_MODEL_PATH must name the checkpoint directory}"
: "${NEMOTRON_SIDECAR_PORT:=8000}"

VENV=/opt/nemo-speech/venv
# The slim server image has no CUDA toolkit. FlashInfer JIT-compiles its fused-MoE kernels at
# first start with nvcc and includes nvrtc.h, so point it at the CUDA 13.0 wheels the build
# installed into the venv (nvcc, crt, nvrtc, runtime all unpack under nvidia/cu13). The venv's
# bin/ also has the ninja FlashInfer runs its JIT build with.
export CUDA_HOME="${VENV}/lib/python3.13/site-packages/nvidia/cu13"
export PATH="${VENV}/bin:${CUDA_HOME}/bin:${PATH}"

# The cold JIT warmup takes ~20 min; b10cache (/cache/org) keeps it across replicas and
# restarts when enabled, else fall back to local disk. Separate from the batch preset's
# directory: this toolchain comes from wheels, not the image.
CACHE_BASE=/cache/org/nemotron-3.5-transcribe-30b-a4b-streaming
mkdir -p "${CACHE_BASE}" 2>/dev/null && [[ -w "${CACHE_BASE}" ]] || CACHE_BASE=/root/.cache/nemotron-3.5-transcribe-30b-a4b-streaming
export FLASHINFER_WORKSPACE_BASE="${CACHE_BASE}"
export VLLM_CACHE_ROOT="${CACHE_BASE}/vllm"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_BASE}/inductor"
export TRITON_CACHE_DIR="${CACHE_BASE}/triton"
mkdir -p "${VLLM_CACHE_ROOT}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

# config.json names pretrained_llm as the relative path "llm_backbone".
cd "${BASETEN_MODEL_PATH}"

# The card's --mamba-cache-mode all resolves to none: vLLM 0.28.0 keeps prefix caching off for
# this model. Deviation from the card: thinking off by default (the backend also sends it off).
# The public config.json renamed the architecture; the pinned Speech commit registers only the
# old name.
exec "${VENV}/bin/vllm" serve "${BASETEN_MODEL_PATH}" \
    --tokenizer "${BASETEN_MODEL_PATH}" \
    --served-model-name nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B \
    --host 127.0.0.1 \
    --port "${NEMOTRON_SIDECAR_PORT}" \
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
    --reasoning-parser nemotron_v3
