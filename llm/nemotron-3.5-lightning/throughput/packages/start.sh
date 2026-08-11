#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="/models/nemotron-3.5-lightning-nvfp4"
DRAFT_MODEL_DIR="/models/nemotron-3.5-lightning-nvfp4-dflash"

if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
    echo "FATAL: public NVIDIA NVFP4 verifier is not mounted at ${MODEL_DIR}." >&2
    echo "Check the verifier weights source, pinned revision, hf_access_token secret, and BDN mount status." >&2
    exit 1
fi

if [[ ! -f "${MODEL_DIR}/model.safetensors.index.json" ]]; then
    echo "FATAL: public NVIDIA NVFP4 verifier index is missing at ${MODEL_DIR}." >&2
    echo "Check the verifier allow_patterns and BDN download status." >&2
    exit 1
fi

if [[ ! -f "${DRAFT_MODEL_DIR}/config.json" ]]; then
    echo "FATAL: public NVIDIA DFlash drafter is not mounted at ${DRAFT_MODEL_DIR}." >&2
    echo "Check the drafter weights source, pinned revision, hf_access_token secret, and BDN mount status." >&2
    exit 1
fi

if [[ ! -f "${DRAFT_MODEL_DIR}/model.safetensors" ]]; then
    echo "FATAL: public NVIDIA DFlash drafter weights are missing." >&2
    echo "Check the drafter allow_patterns and BDN download status." >&2
    exit 1
fi

if [[ ! -f "${DRAFT_MODEL_DIR}/hf_quant_config.json" ]]; then
    echo "FATAL: public NVIDIA DFlash W4A16 quantization metadata is missing." >&2
    echo "Check the drafter allow_patterns and BDN download status." >&2
    exit 1
fi

# This is the verified public-HF 1M-context configuration. Explicit async
# scheduling is incompatible with DFlash in this vLLM release. H100 leaves
# Mamba SSU algorithm selection automatic because explicit vertical/horizontal
# MTP kernels require SM100+.
exec vllm serve "${MODEL_DIR}" \
    --served-model-name nvidia/nemotron-3.5-lightning-nvfp4 \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --max-num-seqs 512 \
    --max-model-len 1048576 \
    --max-num-batched-tokens 32768 \
    --enable-prefix-caching \
    --mamba-cache-mode align \
    --quantization modelopt_fp4 \
    --speculative-config "{\"method\":\"dflash\",\"model\":\"${DRAFT_MODEL_DIR}\",\"num_speculative_tokens\":3}" \
    --moe-backend humming \
    --linear-backend humming \
    --mamba-backend flashinfer \
    --mamba-ssm-cache-dtype float16 \
    --enable-mamba-cache-stochastic-rounding \
    --mamba-cache-philox-rounds 5
