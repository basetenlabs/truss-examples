#!/usr/bin/env bash
# Every serve flag is an environment variable (set in config.yaml) so /batch-bench can sweep it
# with truss_overrides.environment_variables. Fallbacks match config.yaml (64k context, 12,288 KV blocks, CUDA graphs).
set -e

MODEL_PATH="${BASETEN_MODEL_PATH:-/models/vibevoice-asr}"
SERVED_MODEL="${VV_SERVED_MODEL:-vibevoice}"
MAX_NUM_SEQS="${VV_MAX_NUM_SEQS:-16}"
MAX_MODEL_LEN="${VV_MAX_MODEL_LEN:-65536}"
GPU_MEM_UTIL="${VV_GPU_MEMORY_UTILIZATION:-0.85}"
NUM_GPU_BLOCKS="${VV_NUM_GPU_BLOCKS:-12288}"   # 0 or empty = let vLLM size the KV pool
ENFORCE_EAGER="${VV_ENFORCE_EAGER:-0}"        # 1 = eager, no CUDA graphs
PROXY="${VV_PROXY:-1}"                        # 0 = vLLM on the public port, no proxy

echo "[entrypoint] Applying microsoft/VibeVoice plugin patches..."
python3 /app/data/patch.py

# The plugin writes tokenizer files next to the weights. A read-only checkpoint mount (training
# checkpoints) gets a writable directory of symlinks instead.
SERVE_PATH="$MODEL_PATH"
if [ ! -w "$MODEL_PATH" ]; then
  SERVE_PATH=/tmp/vibevoice-model
  mkdir -p "$SERVE_PATH"
  for f in "$MODEL_PATH"/*; do ln -sf "$f" "$SERVE_PATH/$(basename "$f")"; done
fi
echo "[entrypoint] Generating tokenizer files in $SERVE_PATH..."
python3 -m vllm_plugin.tools.generate_tokenizer_files --output "$SERVE_PATH"

# Stop tokens. The checkpoint ships no generation_config.json, so vLLM stops only on the tokenizer EOS
# <|endoftext|> (151643) while the model ends its turn with <|im_end|> (151645) and runs on to max_tokens.
# vLLM adds every eos_token_id listed here to each request's stop tokens, on every route and with
# VV_PROXY=0. An existing file (a fine-tune's) keeps its other fields and gains the two ids.
echo "[entrypoint] Writing eos_token_id [151643, 151645] to $SERVE_PATH/generation_config.json..."
python3 - "$SERVE_PATH/generation_config.json" <<'PY'
import json, os, sys

path = sys.argv[1]
cfg = {}
if os.path.exists(path):
    with open(path) as f:
        cfg = json.load(f)
if os.path.islink(path):
    os.unlink(path)  # read-only checkpoint: replace the symlink with a file
eos = cfg.get("eos_token_id")
eos = [] if eos is None else [eos] if isinstance(eos, int) else list(eos)
cfg["eos_token_id"] = eos + [t for t in (151643, 151645) if t not in eos]
with open(path, "w") as f:
    json.dump(cfg, f, indent=2)
print(f"[entrypoint] generation_config.json: {json.dumps(cfg)}")
PY

args=(
  --served-model-name "$SERVED_MODEL"
  --trust-remote-code
  --dtype bfloat16
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --no-enable-prefix-caching
  --enable-chunked-prefill
  --chat-template-content-format openai
  --allowed-local-media-path /app
  --media-io-kwargs '{"audio": {"target_sr": 24000}}'
  --skip-mm-profiling
)
if [ -n "$NUM_GPU_BLOCKS" ] && [ "$NUM_GPU_BLOCKS" != "0" ]; then
  args+=(--num-gpu-blocks-override "$NUM_GPU_BLOCKS")
fi
if [ "$ENFORCE_EAGER" = "1" ]; then
  args+=(--enforce-eager)
fi
if [ -n "${VV_MAX_NUM_BATCHED_TOKENS:-}" ]; then
  args+=(--max-num-batched-tokens "$VV_MAX_NUM_BATCHED_TOKENS")
fi
# Opt-in: disable the multimodal processor cache (repeated-clip cells). Unset = vLLM default.
if [ -n "${VV_MM_PROCESSOR_CACHE_GB:-}" ]; then
  args+=(--mm-processor-cache-gb "$VV_MM_PROCESSOR_CACHE_GB")
fi
# shellcheck disable=SC2206
extra=(${VV_EXTRA_VLLM_ARGS:-})

if [ "$PROXY" != "1" ]; then
  echo "[entrypoint] Starting vLLM serve on :8000 (no proxy): ${args[*]} ${extra[*]}"
  exec vllm serve "$SERVE_PATH" "${args[@]}" "${extra[@]}" --host 0.0.0.0 --port 8000
fi

echo "[entrypoint] Starting vLLM serve on 127.0.0.1:8001: ${args[*]} ${extra[*]}"
vllm serve "$SERVE_PATH" "${args[@]}" "${extra[@]}" --host 127.0.0.1 --port 8001 &
VLLM_PID=$!

echo "[entrypoint] Starting vv_longform proxy on :8000"
VV_UPSTREAM=http://127.0.0.1:8001 VV_MAX_MODEL_LEN="$MAX_MODEL_LEN" VV_SERVED_MODEL="$SERVED_MODEL" \
  PYTHONPATH=/packages python3 -m vv_longform &
PROXY_PID=$!

# If either process exits, stop the other and exit so the replica restarts.
trap 'kill $VLLM_PID $PROXY_PID 2>/dev/null' TERM INT
set +e
wait -n $VLLM_PID $PROXY_PID
code=$?
echo "[entrypoint] a child exited (code $code); stopping"
kill $VLLM_PID $PROXY_PID 2>/dev/null
wait
exit $code
