"""Call a Triton TensorRT-LLM ensemble deployment on Baseten via /predict."""

import os

import httpx

MODEL_ID = os.environ.get("BASETEN_MODEL_ID", "YOUR_MODEL_ID")
API_KEY = os.environ["BASETEN_API_KEY"]

PREDICT_URL = (
    f"https://model-{MODEL_ID}.api.baseten.co/environments/production/predict"
)

payload = {
    "text_input": "What is machine learning?",
    "max_tokens": 64,
    "bad_words": "",
    "stop_words": "",
}

with httpx.Client(timeout=300.0) as client:
    response = client.post(
        PREDICT_URL,
        json=payload,
        headers={"Authorization": f"Api-Key {API_KEY}"},
    )
    response.raise_for_status()
    print(response.json())
