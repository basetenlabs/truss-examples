"""Call a Triton vLLM deployment via the OpenAI-compatible API on Baseten."""

import os

from openai import OpenAI

MODEL_ID = os.environ.get("BASETEN_MODEL_ID", "YOUR_MODEL_ID")
API_KEY = os.environ["BASETEN_API_KEY"]
SERVED_MODEL = os.environ.get("SERVED_MODEL", "llama-3.2-1b-instruct")

client = OpenAI(
    base_url=(
        f"https://model-{MODEL_ID}.api.baseten.co/environments/production/sync/v1"
    ),
    api_key=API_KEY,
)

response = client.chat.completions.create(
    model=SERVED_MODEL,
    messages=[
        {
            "role": "user",
            "content": "What is NVIDIA Triton Inference Server?",
        }
    ],
    max_tokens=128,
    temperature=0.0,
)

print(response.choices[0].message.content)
