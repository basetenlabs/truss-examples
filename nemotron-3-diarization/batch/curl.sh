#!/usr/bin/env bash
# Diarize a recording by URL with the Nemotron 3 Diarization batch endpoint.
#   export BASETEN_API_KEY=... MODEL_ID=...
#   ./curl.sh https://example.com/meeting.wav [offline|low|ultralow]
set -euo pipefail
URL=${1:?audio url}; LATENCY=${2:-offline}
curl -sS -X POST "https://model-${MODEL_ID}.api.baseten.co/environments/production/predict" \
  -H "Authorization: Api-Key ${BASETEN_API_KEY}" \
  -H "Content-Type: application/json" \
  -d "{\"diarization_input\": {\"audio\": {\"url\": \"${URL}\"}, \"latency\": \"${LATENCY}\"}}"
echo
