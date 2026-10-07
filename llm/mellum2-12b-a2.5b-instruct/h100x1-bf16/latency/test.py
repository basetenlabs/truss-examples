#!/usr/bin/env python3
"""
Test script for Mellum2-12B-A2.5B-Instruct deployed on Baseten.

Usage:
    BASETEN_API_KEY=<key> BASETEN_MODEL_ID=<model_id> python test.py

Tests two scenarios:
  1. Streaming chat completion — code generation
  2. Tool calling             — function calling with hermes parser
"""

import json
import os
import sys
import requests

# ── config ────────────────────────────────────────────────────────────────────

API_KEY  = os.environ.get("BASETEN_API_KEY")
MODEL_ID = os.environ.get("BASETEN_MODEL_ID")

if not API_KEY or not MODEL_ID:
    print("Error: set BASETEN_API_KEY and BASETEN_MODEL_ID environment variables.")
    sys.exit(1)

BASE_URL   = f"https://model-{MODEL_ID}.api.baseten.co/environments/production/sync/v1"
CHAT_URL   = BASE_URL + "/chat/completions"
MODEL_NAME = "JetBrains/Mellum2-12B-A2.5B-Instruct"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
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
            {"role": "user", "content": "Write a Python function to reverse a string."}
        ],
        "stream": True,
        "max_tokens": 512,
        "temperature": 0.6,
        "top_p": 0.95,
    },
    stream=True,
    timeout=120,
)

if not resp.ok:
    print(f"   Error {resp.status_code}: {resp.text}")
    failures.append(f"Streaming: HTTP {resp.status_code}")
else:
    chunks = []
    for line in resp.iter_lines():
        if not line or line == b"data: [DONE]":
            continue
        if line.startswith(b"data: "):
            try:
                delta = json.loads(line[6:])["choices"][0]["delta"]
                c = delta.get("content", "")
                if c:
                    chunks.append(c)
                    print(c, end="", flush=True)
            except (json.JSONDecodeError, KeyError):
                pass
    print()
    content = "".join(chunks)
    if len(content) < 20:
        failures.append(f"Streaming: response too short ({len(content)} chars)")
    else:
        print(f"   OK ({len(content)} chars)")

# ── 2. tool calling ───────────────────────────────────────────────────────────

print()
print("=" * 60)
print("2. Tool calling (hermes parser)")

tools = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Execute a Python code snippet and return its stdout.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute.",
                    }
                },
                "required": ["code"],
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
            {"role": "user", "content": "Use run_python to print the first 5 Fibonacci numbers."}
        ],
        "tools": tools,
        "tool_choice": "auto",
        "stream": False,
        "max_tokens": 256,
        "temperature": 0.6,
    },
    timeout=120,
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
        print(f"   Tool called: {fn_name}({json.dumps(fn_args)})")
        if fn_name != "run_python":
            failures.append(f"Tool calling: expected 'run_python', got '{fn_name}'")
        elif "code" not in fn_args:
            failures.append("Tool calling: 'code' argument missing")
        else:
            print("   OK")
    else:
        content = choice["message"].get("content", "")
        print(f"   No tool call (finish_reason={choice.get('finish_reason')!r}). Content: {content[:120]!r}")
        failures.append("Tool calling: expected finish_reason='tool_calls'")

# ── summary ───────────────────────────────────────────────────────────────────

print()
if failures:
    print("FAIL")
    for msg in failures:
        print(f"  ✗ {msg}")
    sys.exit(1)
else:
    print("PASS")
