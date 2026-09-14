#!/usr/bin/env bash
set -e
echo "[entrypoint] Applying microsoft/VibeVoice plugin patches..."
python3 /app/data/patch.py
echo "[entrypoint] Generating tokenizer files..."
python3 -m vllm_plugin.tools.generate_tokenizer_files --output /models/vibevoice-asr
echo "[entrypoint] Starting vLLM serve..."
exec vllm serve /models/vibevoice-asr \
  --served-model-name vibevoice \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-num-seqs 16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --num-gpu-blocks-override 4096 \
  --no-enable-prefix-caching \
  --enable-chunked-prefill \
  --chat-template-content-format openai \
  --allowed-local-media-path /app \
  --media-io-kwargs "{\"audio\": {\"target_sr\": 24000}}" \
  --enforce-eager \
  --skip-mm-profiling \
  --host 0.0.0.0 \
  --port 8000
