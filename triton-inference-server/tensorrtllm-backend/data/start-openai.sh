#!/usr/bin/env bash
# Prepare TensorRT-LLM engines and start Triton's OpenAI-compatible frontend.
set -euo pipefail

MODEL_REPO="/app/data/model_repository"
ENGINE_SRC="${ENGINE_DIR:-/models/engine}"
TOKENIZER_DIR="${TOKENIZER_DIR:-/models/tokenizer}"
ENGINE_DEST="${MODEL_REPO}/tensorrt_llm/1"

mkdir -p "${ENGINE_DEST}"

if [ -d "${ENGINE_SRC}" ] && [ -n "$(ls -A "${ENGINE_SRC}" 2>/dev/null || true)" ]; then
  echo "Copying TensorRT-LLM engine from ${ENGINE_SRC} to ${ENGINE_DEST}"
  cp -a "${ENGINE_SRC}/." "${ENGINE_DEST}/"
else
  echo "WARNING: No TensorRT-LLM engine found at ${ENGINE_SRC}."
  echo "Upload a built engine via weights (BDN) before deploying."
fi

export triton_tokenizer_repository="${TOKENIZER_DIR}"

if [ -f /secrets/hf_access_token ]; then
  export HF_TOKEN
  HF_TOKEN="$(cat /secrets/hf_access_token)"
fi

cd /opt/tritonserver/python/openai
exec python3 openai_frontend/main.py \
  --model-repository "${MODEL_REPO}" \
  --tokenizer "${TOKENIZER_DIR}" \
  --backend tensorrtllm \
  --host 0.0.0.0 \
  --openai-port 8000
