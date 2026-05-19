#!/usr/bin/env bash
# Start Triton's OpenAI-compatible frontend over the TRT-LLM LLMAPI (PyTorch) model.
set -euo pipefail

if [ -f /secrets/hf_access_token ]; then
  export HF_TOKEN
  HF_TOKEN="$(cat /secrets/hf_access_token)"
fi

cd /opt/tritonserver/python/openai
exec python3 openai_frontend/main.py \
  --model-repository /app/data/model_repository \
  --tokenizer /models/qwen \
  --backend tensorrtllm \
  --host 0.0.0.0 \
  --openai-port 8000
