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
# The slim server image has no CUDA toolkit. FlashInfer JIT-compiles its ReplaySSM module at
# start with nvcc and includes nvrtc.h (fused-MoE and sampling come prebuilt from
# flashinfer-jit-cache), so point it at the CUDA 13.0 wheels in the venv (nvcc,
# crt, nvrtc, runtime all unpack under nvidia/cu13). The venv's bin/ also has the ninja
# FlashInfer runs its JIT build with.
export CUDA_HOME="${VENV}/lib/python3.13/site-packages/nvidia/cu13"
export PATH="${VENV}/bin:${CUDA_HOME}/bin:${PATH}"

# As the batch preset: the BDN startup cache (ORG_ENABLE_BDN_STARTUP_CACHE) sets the JIT cache
# paths (VLLM_CACHE_ROOT, TORCHINDUCTOR_CACHE_DIR, TRITON_CACHE_DIR, FLASHINFER_WORKSPACE_BASE,
# ...) and restores them across replicas and restarts, keyed by GPU type.
# Never point them at /cache/org: on that shared mount the engine sat idle after "Warming up
# Mamba2" until the startup check killed it (#1077).

# config.json names pretrained_llm as the relative path "llm_backbone".
cd "${BASETEN_MODEL_PATH}"

# The batch preset's card flags for vLLM 0.32.0: ReplaySSM with the FlashInfer Mamba backend
# and prefix caching off. Deviation from the card: thinking off by default (the backend also
# sends it off). NeMo Speech main registers the current architecture name, so no
# --hf-overrides.
exec "${VENV}/bin/vllm" serve "${BASETEN_MODEL_PATH}" \
    --tokenizer "${BASETEN_MODEL_PATH}" \
    --served-model-name nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B \
    --host 127.0.0.1 \
    --port "${NEMOTRON_SIDECAR_PORT}" \
    --trust-remote-code \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.94 \
    --max-model-len 65536 \
    --max-num-seqs 192 \
    --max-num-batched-tokens 4096 \
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
    --reasoning-parser nemotron_v3
