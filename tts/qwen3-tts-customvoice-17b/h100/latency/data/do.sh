#!/bin/bash
set -euo pipefail

# Start command for the Qwen3-TTS CustomVoice (preset speakers) vLLM-Omni
# server. Invoked by config.yaml's docker_server.start_command as:
# /app/data/do.sh /models/Qwen3-TTS-12Hz-1.7B-CustomVoice

MODEL_PATH="${1:-/models/Qwen3-TTS-12Hz-1.7B-CustomVoice}"
DEPLOY_CONFIG="/app/data/qwen3_tts_deploy.yaml"
PORT="8091"
WARMUP_TEXT="${WARMUP_TEXT:-Hello, this is a warmup request to trace the CUDA graphs.}"
WARMUP_VOICE="${WARMUP_VOICE:-vivian}"
WARMUP_ITERS="${WARMUP_ITERS:-2}"

cd /app/vllm-omni

# Restore a shared torch.compile cache before vLLM-Omni starts. On a cache
# miss, this process saves the artifacts produced during server initialization
# so subsequent replicas can skip most of the JIT work.
echo "Starting shared torch compile cache manager..."
b10-compile-cache &

echo "Starting Qwen3-TTS server: model=${MODEL_PATH} deploy_config=${DEPLOY_CONFIG} port=${PORT}"
vllm-omni serve "${MODEL_PATH}" \
    --deploy-config "${DEPLOY_CONFIG}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --trust-remote-code \
    --omni &
SERVER_PID=$!

echo "Waiting for vllm server on port ${PORT}..."
until curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "vllm server process exited during startup." >&2
        wait "${SERVER_PID}"
        exit 1
    fi
    sleep 2
done
echo "vllm server is ready on port ${PORT}."

# Synthesize a few utterances so the first user request doesn't pay for
# lazy first-request work.
if [[ "${WARMUP_ITERS}" -gt 0 ]]; then
    echo "Warming up model with voice '${WARMUP_VOICE}' (${WARMUP_ITERS} iteration(s))..."
    for i in $(seq 1 "${WARMUP_ITERS}"); do
        if curl -s -f -X POST "http://localhost:${PORT}/v1/audio/speech" \
            -H "Content-Type: application/json" \
            -d "{\"input\": \"${WARMUP_TEXT}\", \"voice\": \"${WARMUP_VOICE}\", \"task_type\": \"CustomVoice\", \"stream\": true, \"response_format\": \"pcm\"}" \
            --no-buffer -o /dev/null; then
            echo "  Warmup iteration ${i} complete."
        else
            echo "  Warmup iteration ${i} failed — continuing."
        fi
    done
    echo "Model warmup complete."
fi

# Hand the foreground back to the server.
wait "${SERVER_PID}"
