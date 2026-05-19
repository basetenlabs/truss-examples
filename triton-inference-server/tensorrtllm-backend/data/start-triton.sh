#!/usr/bin/env bash
# Prepare the TensorRT-LLM Triton model repository and start tritonserver.
set -euo pipefail

MODEL_REPO="/app/data/model_repository"
ENGINE_SRC="${ENGINE_DIR:-/models/engine}"
TOKENIZER_DIR="${TOKENIZER_DIR:-/models/tokenizer}"
ENGINE_DEST="${MODEL_REPO}/tensorrt_llm/1"

mkdir -p "${ENGINE_DEST}"

if [ -d "${ENGINE_SRC}" ] && [ -n "$(ls -A "${ENGINE_SRC}" 2>/dev/null || true)" ]; then
  echo "Linking TensorRT-LLM engine from ${ENGINE_SRC} to ${ENGINE_DEST}"
  cp -a "${ENGINE_SRC}/." "${ENGINE_DEST}/"
else
  echo "WARNING: No TensorRT-LLM engine found at ${ENGINE_SRC}."
  echo "Upload a built engine via weights (BDN) before deploying."
fi

export triton_tokenizer_repository="${TOKENIZER_DIR}"

if [ -f /secrets/hf_access_token ]; then
  export HUGGING_FACE_HUB_TOKEN
  HUGGING_FACE_HUB_TOKEN="$(cat /secrets/hf_access_token)"
fi

exec tritonserver \
  --model-repository="${MODEL_REPO}" \
  --http-port=8000 \
  --grpc-port=8001 \
  --metrics-port=8002 \
  --allow-http=true \
  --allow-grpc=true
