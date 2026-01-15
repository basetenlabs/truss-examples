#!/bin/bash
set -e
export VLLM_USE_V1=1
export HF_TOKEN=$(cat /secrets/hf_access_token)

echo "=== Checking bptr manifest ==="
ls -la /static-bptr/ 2>/dev/null || echo "No /static-bptr directory"
ls -la /bptr/ 2>/dev/null || echo "No /bptr directory"

echo "=== Running truss_transfer ==="
python -c '
import truss_transfer
import os

print("Checking manifest paths...")
for p in ["/static-bptr/static-bptr-manifest.json", "/bptr/bptr-manifest", "/bptr/bptr-manifest.json"]:
    print(f"  {p}: exists={os.path.exists(p)}")

print("Running lazy_data_resolve...")
result = truss_transfer.lazy_data_resolve("/app/model_cache")
print(f"Result: {result}")
'

echo "=== Contents of /app/model_cache ==="
find /app/model_cache -type f | head -20 || echo "model_cache empty or missing"

echo "=== Starting vLLM ==="
exec vllm serve /app/model_cache/gemma \
  --served-model-name gemma \
  --max-num-seqs 8 \
  --max-model-len 16384 \
  --limit_mm_per_prompt 'image=1' \
  --hf-overrides '{"do_pan_and_scan": true}' \
  --gpu-memory-utilization 0.95

