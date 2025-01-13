import base64

import httpx

DEPLOYMENT_URL = ""
API_KEY = ""
# Create client for connection reuse
with httpx.Client() as client:
    # Make the API request
    resp = client.post(
        DEPLOYMENT_URL,
        headers={"Authorization": f"Api-Key {API_KEY}"},
        json={"text": "Hello world" * 32, "voice": "af", "speed": 1.0},
        timeout=None,
    )

# Get the base64 encoded audio
response_data = resp.json()
audio_base64 = response_data["base64"]

# Decode the base64 string
audio_bytes = base64.b64decode(audio_base64)

# Write to a WAV file
with open("output.wav", "wb") as f:
    f.write(audio_bytes)

print("Audio saved to output.wav")
