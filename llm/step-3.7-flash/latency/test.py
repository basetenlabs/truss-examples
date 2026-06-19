#!/usr/bin/env python3
"""
Test script for Step-3.7-Flash (FP8) deployed on Baseten.

Usage:
    BASETEN_API_KEY=<key> BASETEN_MODEL_ID=<model_id> BASETEN_DEPLOYMENT_ID=<deployment_id> python test.py

Tests three scenarios:
  1. Streaming chat completion
  2. Tool calling             — step3p5 parser
  3. Multimodal image input   — vision-language
"""

import json
import os
import sys
import requests

# ── config ────────────────────────────────────────────────────────────────────

API_KEY       = os.environ.get("BASETEN_API_KEY")
MODEL_ID      = os.environ.get("BASETEN_MODEL_ID")
DEPLOYMENT_ID = os.environ.get("BASETEN_DEPLOYMENT_ID")

if not API_KEY or not MODEL_ID or not DEPLOYMENT_ID:
    print("Error: set BASETEN_API_KEY, BASETEN_MODEL_ID, and BASETEN_DEPLOYMENT_ID environment variables.")
    sys.exit(1)

PREDICT_URL = f"https://model-{MODEL_ID}.api.baseten.co/deployment/{DEPLOYMENT_ID}/predict"
CHAT_URL    = PREDICT_URL
MODEL_NAME  = "stepfun-ai/Step-3.7-Flash-FP8"

HEADERS = {
    "Authorization": f"Api-Key {API_KEY}",
    "Content-Type": "application/json",
}

failures = []

# ── 1. streaming chat completion ──────────────────────────────────────────────

print("=" * 60)
print("1. Streaming chat completion")

resp = requests.post(
    CHAT_URL,
    headers=HEADERS,
    json={
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": "Write a Python function to compute the nth Fibonacci number iteratively."}
        ],
        "stream": True,
        "max_tokens": 1024,
        "temperature": 1.0,
    },
    stream=True,
    timeout=300,
)

if not resp.ok:
    print(f"   Error {resp.status_code}: {resp.text}")
    failures.append(f"Streaming: HTTP {resp.status_code}")
else:
    reasoning_chunks, content_chunks = [], []
    for line in resp.iter_lines():
        if not line or line == b"data: [DONE]":
            continue
        if line.startswith(b"data: "):
            try:
                chunk = json.loads(line[6:])
                delta = chunk["choices"][0]["delta"]
                r = delta.get("reasoning_content", "")
                c = delta.get("content", "")
                if r:
                    reasoning_chunks.append(r)
                if c:
                    content_chunks.append(c)
                    print(c, end="", flush=True)
            except (json.JSONDecodeError, KeyError):
                pass
    print()
    reasoning = "".join(reasoning_chunks)
    content   = "".join(content_chunks)
    print(f"   reasoning_content: {len(reasoning)} chars, content: {len(content)} chars")
    if len(content) < 20:
        failures.append(f"Streaming: response too short ({len(content)} chars)")
    else:
        print("   OK")

# ── 2. tool calling ───────────────────────────────────────────────────────────

print()
print("=" * 60)
print("2. Tool calling (step3p5 parser)")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and country, e.g. 'San Francisco, US'",
                    }
                },
                "required": ["location"],
            },
        },
    }
]

resp2 = requests.post(
    CHAT_URL,
    headers=HEADERS,
    json={
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": "What's the weather like in Tokyo right now?"}
        ],
        "tools": tools,
        "tool_choice": "auto",
        "stream": False,
        "max_tokens": 256,
        "temperature": 1.0,
    },
    timeout=300,
)

if not resp2.ok:
    print(f"   Error {resp2.status_code}: {resp2.text}")
    failures.append(f"Tool calling: HTTP {resp2.status_code}")
else:
    choice = resp2.json()["choices"][0]
    tool_calls = choice["message"].get("tool_calls") or []

    if choice.get("finish_reason") == "tool_calls" and tool_calls:
        tc = tool_calls[0]
        fn_name = tc["function"]["name"]
        try:
            fn_args = json.loads(tc["function"]["arguments"])
        except json.JSONDecodeError:
            fn_args = {}
        print(f"   Tool called: {fn_name}({fn_args})")
        if fn_name != "get_weather":
            failures.append(f"Tool calling: expected 'get_weather', got '{fn_name}'")
        elif "location" not in fn_args:
            failures.append("Tool calling: 'location' argument missing")
        else:
            print("   OK")
    else:
        content = choice["message"].get("content", "")
        print(f"   No tool call (finish_reason={choice.get('finish_reason')!r}). Content: {content[:120]!r}")
        failures.append("Tool calling: expected finish_reason='tool_calls'")

# ── 3. multimodal image input ─────────────────────────────────────────────────

print()
print("=" * 60)
print("3. Multimodal image input")

resp3 = requests.post(
    CHAT_URL,
    headers=HEADERS,
    json={
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
                        },
                    },
                    {"type": "text", "text": "What landmark is shown in this image? Reply in one sentence."},
                ],
            }
        ],
        "stream": False,
        "max_tokens": 128,
        "temperature": 1.0,
    },
    timeout=300,
)

if not resp3.ok:
    print(f"   Error {resp3.status_code}: {resp3.text}")
    failures.append(f"Multimodal: HTTP {resp3.status_code}")
else:
    content = resp3.json()["choices"][0]["message"].get("content") or ""
    print(f"   Response: {content}")
    if len(content) < 10:
        failures.append(f"Multimodal: response too short ({len(content)} chars)")
    elif "liberty" not in content.lower() and "statue" not in content.lower():
        failures.append(f"Multimodal: expected mention of Statue of Liberty, got: {content[:120]!r}")
    else:
        print("   OK")

# ── summary ───────────────────────────────────────────────────────────────────

print()
if failures:
    print("FAIL")
    for msg in failures:
        print(f"  ✗ {msg}")
    sys.exit(1)
else:
    print("PASS")
