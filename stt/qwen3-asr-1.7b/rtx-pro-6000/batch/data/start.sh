#!/usr/bin/env bash
# Patch the installed vLLM in place (data/patch.py), then serve.
set -euo pipefail
echo "[start.sh] applying b10 patch"
python3 /app/data/patch.py
echo "[start.sh] starting vllm serve"
# --api-server-count 4: the route decodes and chunks each upload on the CPU in the
#   API-server process, which caps throughput at one process; four lifts the ceiling
#   ~60% at high concurrency on this 16 vCPU instance.
# --attention-backend FLASHINFER --kv-cache-dtype fp8: an FP8 KV cache halves the
#   bytes attention reads per decode step (attention is ~half of GPU time on fresh
#   audio). FlashAttention cannot run an FP8 cache on this GPU (SM120); FlashInfer can.
#   Weights stay bf16: FP8 weights cost 0.1-0.3 pt WER for no throughput gain.
# --override-generation-config: default output cap per generated chunk when the
#   caller sets none. Non-speech audio with a forced language otherwise decodes
#   until the context limit (minutes); 4096 tokens is far above any 30 s chunk's
#   transcript and callers can still pass their own max_completion_tokens.
exec vllm serve "$BASETEN_MODEL_PATH" \
  --tensor-parallel-size 1 \
  --served-model-name Qwen/Qwen3-ASR-1.7B \
  --gpu-memory-utilization 0.8 \
  --host 0.0.0.0 \
  --port 8000 \
  --load-format runai_streamer \
  --api-server-count 4 \
  --attention-backend FLASHINFER \
  --kv-cache-dtype fp8 \
  --override-generation-config.max_new_tokens 4096
