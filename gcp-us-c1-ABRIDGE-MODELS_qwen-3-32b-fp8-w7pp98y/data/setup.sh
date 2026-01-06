#!/bin/bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export NO_PROXY="*"

# Install dependencies
apt-get update && apt-get install -y curl gnupg
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor | tee /usr/share/keyrings/cloud.google.gpg > /dev/null
apt-get update && apt-get install -y google-cloud-sdk
gcloud auth activate-service-account --key-file=/secrets/abridge-models-read-sa-key

# Note: Datadog Agent will start after model download, just before vLLM starts

# Setup directories
CACHE_DIR="/cache/org"
MODEL_DIR="$CACHE_DIR/Qwen/Qwen3-32B-FP8"
MODEL_PATH="$MODEL_DIR/model"
LORAS_PATH="$MODEL_DIR/loras"
mkdir -p "$MODEL_PATH" "$LORAS_PATH"

# Function to get file lists (GCP or local)
get_files() {
    local path="$1"
    if [[ "$path" == gs://* ]]; then
        # GCP path
        local base_path=$(echo "$path" | sed 's|gs://[^/]*/||')
        gsutil ls -r "$path" 2>/dev/null | sed 's|gs://[^/]*/||' | sed "s|^$base_path||" | sed 's|^/||' | sed 's|^\./||' | grep -v '^\.cache/' | grep -v '/$' | grep -v ':$' | sort
    else
        # Local path
        [ -d "$path" ] && find "$path" -type f 2>/dev/null | sed "s|^$path/||" | sed 's|^\./||' | grep -v '^\.cache/' | sort || echo ""
    fi
}

# Check if files exist in cache
GCP_MODEL="gs://abridge-artifact-registry-abridge-models/deployments/Qwen/Qwen3-32B-FP8/model/"
GCP_LORAS="gs://abridge-artifact-registry-abridge-models/deployments/Qwen/Qwen3-32B-FP8/loras/"

MODEL_FILES_MATCH=$(get_files "$GCP_MODEL" | tr -d '\n' | diff - <(get_files "$MODEL_PATH" | tr -d '\n') >/dev/null && echo "true" || echo "false")

# No LoRAs configured - only check model files
echo "=== CACHE STATUS ==="
echo "Model files: $([ "$MODEL_FILES_MATCH" = "true" ] && echo "✓ Present" || echo "✗ Missing")"
echo "Lora files: N/A (no LoRAs configured)"
echo "==================="

if [ "$MODEL_FILES_MATCH" = "true" ]; then
    echo "Warming up cache..."
    find "$MODEL_DIR" -type f -print0 | xargs -0 -P 16 -I {} dd if="{}" of=/dev/null bs=4M
    echo "Cache warm-up completed"
else
    echo "Downloading from GCP..."
    gsutil -m cp -r "$GCP_MODEL"* "$MODEL_PATH/"
fi

# No LoRAs to process
N_LORAS=0
LORA_MODULES=""
# Start Datadog Agent for vLLM metrics collection (right before vLLM starts)
echo ""
echo "=== STARTING DATADOG AGENT ==="
# Read Datadog API key from Baseten secrets (mounted at /secrets/)
if [ -f "/secrets/datadog-api-key" ]; then
    export DD_API_KEY=$(cat /secrets/datadog-api-key)

    # Set hostname for Datadog Agent (required in containerized environments)
    # Use the DD_SERVICE name as hostname to identify this deployment
    export DD_HOSTNAME="${DD_SERVICE}-baseten-$(hostname || echo 'unknown')"

    # Read Baseten environment from dynamic config file and add to DD_TAGS
    if [ -f "/etc/b10_dynamic_config/environment" ]; then
        # Try to parse as JSON and extract the "name" field
        BASETEN_ENV=$(python3 -c "import json, sys; data=json.load(open('/etc/b10_dynamic_config/environment')); print(data.get('name', ''))" 2>/dev/null || echo "")
        if [ -n "$BASETEN_ENV" ]; then
            export DD_TAGS="${DD_TAGS} baseten_environment:${BASETEN_ENV}"
            echo "✓ Added baseten_environment tag: $BASETEN_ENV"
        else
            echo "⚠ Baseten environment file exists but could not parse JSON or extract 'name' field"
        fi
    else
        echo "⚠ Baseten environment file not found at /etc/b10_dynamic_config/environment"
    fi

    echo "✓ Starting Datadog Agent for metrics collection"
    echo "  Site: $DD_SITE | Service: $DD_SERVICE | Env: $DD_ENV"
    echo "  Tags: $DD_TAGS"

    # Start Datadog Agent in the background
    # LD_LIBRARY_PATH and PATH already set in Dockerfile ENV
    nohup /opt/datadog-agent/bin/agent/agent run > /var/log/datadog/agent-startup.log 2>&1 &
    AGENT_PID=$!

    echo "  Agent PID: $AGENT_PID"

    # Background health check - verify agent is working after vLLM starts
    (
        sleep 200  # Wait for vLLM to start

        echo ""
        echo "=== Datadog Agent Health Check ==="

        # Check agent process
        if ps -p $AGENT_PID > /dev/null 2>&1; then
            echo "✓ Agent process running (PID: $AGENT_PID)"
        else
            echo "✗ Agent process NOT running!"
            echo "Recent logs:"
            tail -n 30 /var/log/datadog/agent-startup.log 2>/dev/null || echo "No logs"
            echo "==================================="
            exit 0
        fi

        # Check vLLM metrics endpoint
        if curl -s http://localhost:8000/metrics > /dev/null 2>&1; then
            VLLM_METRICS=$(curl -s http://localhost:8000/metrics | grep -c "^vllm" || echo "0")
            echo "✓ vLLM endpoint: $VLLM_METRICS metrics available"
        else
            echo "✗ vLLM endpoint NOT accessible"
        fi

        # Check agent vLLM integration status
        echo ""
        AGENT_STATUS=$(/opt/datadog-agent/bin/agent/agent status 2>&1)
        if echo "$AGENT_STATUS" | grep -q "vllm"; then
            echo "Agent vLLM check:"
            echo "$AGENT_STATUS" | grep -A 10 "vllm"
        else
            echo "⚠ vLLM check not in agent status"
        fi

        # Show recent errors
        echo ""
        echo "Recent errors:"
        tail -n 100 /var/log/datadog/agent-startup.log 2>/dev/null | grep "ERROR" | grep -v "kubelet" | tail -n 5 || echo "No errors"

        echo "==================================="
    ) &

    # Show only critical errors (not warnings or kubelet noise)
    (tail -f /var/log/datadog/agent-startup.log 2>/dev/null | grep --line-buffered "ERROR" | grep -v "kubelet" &)
else
    echo "⚠ Datadog API key not found at /secrets/datadog-api-key"
    echo "  Skipping Datadog Agent startup"
fi
echo "==============================="
echo ""

# Use consistent vLLM arguments generated by the shared configuration
vllm \
serve \
$MODEL_PATH \
--served-model-name \
Qwen/Qwen3-32B-FP8 \
--host \
0.0.0.0 \
--disable-log-requests \
--port \
8000 \
--scheduling-policy \
priority \
--max-model-len \
32768 \
--tensor-parallel-size \
2 \
--enable-prefix-caching \
--max-num-seqs \
128
