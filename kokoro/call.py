import base64
import os
import time

import httpx

DEPLOYMENT_URL = "https://model-w7pp6j0w.api.baseten.co/environments/production/predict"
API_KEY = os.environ.get("BASETEN_API_KEY", "")  # Fill in your Baseten API key or set env var

payload = {
    "text": "Hello, this is a test of Kokoro TTS.",
    "voice": "af_heart",  # See model.py VOICES list for all options
    "speed": 1.0,
}

if not API_KEY:
    raise ValueError("Set your API key: edit call.py or run: export BASETEN_API_KEY=your_key_here")

print(f"Sending request to {DEPLOYMENT_URL}...")
start = time.time()

with httpx.Client() as client:
    resp = client.post(
        DEPLOYMENT_URL,
        headers={"Authorization": f"Api-Key {API_KEY}"},
        json=payload,
        timeout=60,
    )

resp.raise_for_status()
elapsed = time.time() - start
print(f"Response received in {elapsed:.2f}s (status {resp.status_code})")

response_data = resp.json()
audio_bytes = base64.b64decode(response_data["base64"])

output_path = "output.wav"
with open(output_path, "wb") as f:
    f.write(audio_bytes)

print(f"Audio saved to {output_path} ({len(audio_bytes) / 1024:.1f} KB)")
