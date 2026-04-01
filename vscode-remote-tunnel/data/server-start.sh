#!/bin/sh
set -e

cd /app/data
TUNNEL_NAME="$(hostname | head -c 20)"

# Start the FastAPI server in the background with auto-reload on file changes.
# Logs still go to stdout/stderr so they're visible in container logs.
uv run uvicorn server:app --host 0.0.0.0 --port 8090 --reload 2>&1 &

# Wait for server to be ready before proceeding.
printf '\033[32mWaiting for server to start...\033[0m\n'
while ! curl -s http://localhost:8090/health > /dev/null 2>&1; do sleep 0.5; done

# Authenticate with Microsoft for VS Code tunnel access.
# TODO: Use --provider github when GitHub device auth is working reliably.
printf '\033[32mFollow the login prompt below to authenticate the VS Code tunnel.\033[0m\n'
code tunnel user login --provider microsoft

# Print connection URLs.
printf '\033[32mTunnel starting! Connect via:\033[0m\n'
printf '\033[32m  Web:     https://vscode.dev/tunnel/%s/app/data\033[0m\n' "$TUNNEL_NAME"
printf '\033[32m  Desktop: vscode://vscode-remote/tunnel+%s/app/data\033[0m\n' "$TUNNEL_NAME"

# VS Code Server uses its own bundled Node.js TLS stack, ignoring system ca-certificates.
# Point it at the system cert bundle so extension installs don't fail with
# "unable to get local issuer certificate" (see microsoft/vscode#206732).
export NODE_EXTRA_CA_CERTS=/etc/ssl/certs/ca-certificates.crt

# Start the tunnel. The Python extension is preinstalled for connecting clients.
code tunnel --accept-server-license-terms --name "$TUNNEL_NAME" \
    --install-extension ms-python.python 2>&1
