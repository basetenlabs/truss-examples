"""Call a Triton vLLM deployment via the OpenAI-compatible API on Baseten."""

import os
import sys

from openai import OpenAI

MODEL_ID = os.environ.get("BASETEN_MODEL_ID", "YOUR_MODEL_ID")
API_KEY = os.environ.get("BASETEN_API_KEY")
if not API_KEY:
    sys.exit("BASETEN_API_KEY must be set")

STREAM = os.environ.get("STREAM", "").lower() in ("1", "true", "yes")
SERVED_MODEL = os.environ.get("SERVED_MODEL", "qwen3-8b")

client = OpenAI(
    base_url=(
        f"https://model-{MODEL_ID}.api.baseten.co/environments/production/sync/v1"
    ),
    api_key=API_KEY,
)

kwargs = dict(
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

if STREAM:
    stream = client.chat.completions.create(**kwargs, stream=True)
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
    print()
else:
    response = client.chat.completions.create(**kwargs)
    print(response.choices[0].message.content)
