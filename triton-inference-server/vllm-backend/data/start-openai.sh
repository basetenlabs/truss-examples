#!/usr/bin/env bash
set -euo pipefail

if [ -f /secrets/hf_access_token ]; then
  export HF_TOKEN
  HF_TOKEN="$(cat /secrets/hf_access_token)"
fi

cd /opt/tritonserver/python/openai
exec python3 openai_frontend/main.py \
  --model-repository /app/data/model_repository \
  --tokenizer /models/llama \
  --backend vllm \
  --host 0.0.0.0 \
  --openai-port 8000
