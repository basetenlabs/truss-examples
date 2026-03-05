#!/bin/bash
set -e

REPO_DIR="/app/data/vllm-omni"

# Clone if not already present, otherwise pull latest
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/iancarrasco-b10/vllm-omni.git "$REPO_DIR"
fi

cd "$REPO_DIR"
git fetch origin
git pull origin main || true

# Install vllm-omni from the repo
pip install -e .

chmod +x ./examples/online_serving/qwen3_tts/run_server.sh
./examples/online_serving/qwen3_tts/run_server.sh Base 1.7B
