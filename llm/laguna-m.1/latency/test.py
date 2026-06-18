#!/usr/bin/env python3
"""
Test script for Laguna M.1 (FP8) deployed on Baseten.

Usage:
    BASETEN_API_KEY=<key> BASETEN_MODEL_ID=<model_id> python test.py

Tests two scenarios:
  1. Streaming chat completion
  2. Tool calling (poolside_v1 parser)
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

BASE_URL = f"https://model-{MODEL_ID}.api.baseten.co/environments/production/sync/v1"
CHAT_URL = BASE_URL + "/chat/completions"
MODEL    = "poolside/laguna-m.1"

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
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "Write a Python retry wrapper with exponential backoff. Be concise."}
        ],
        "stream": True,
        "temperature": 1.0,
        "max_tokens": 2048,
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
                # Reasoning models stream thinking under reasoning_content
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
    if len(reasoning) + len(content) < 20:
        failures.append(f"Streaming: response too short (reasoning={len(reasoning)}, content={len(content)})")
    else:
        print("   OK")

# ── 2. tool calling ───────────────────────────────────────────────────────────

print()
print("=" * 60)
print("2. Tool calling (poolside_v1 parser)")

tools = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute file path to read"}
                },
                "required": ["path"],
            },
        },
    }
]

resp2 = requests.post(
    CHAT_URL,
    headers=HEADERS,
    json={
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "Read the file at /etc/hostname and tell me its contents."}
        ],
        "tools": tools,
        "tool_choice": "auto",
        "stream": False,
        "temperature": 1.0,
        "max_tokens": 256,
    },
    timeout=300,
)

if not resp2.ok:
    print(f"   Error {resp2.status_code}: {resp2.text}")
    failures.append(f"Tool calling: HTTP {resp2.status_code}")
else:
    result = resp2.json()
    choice = result["choices"][0]
    finish_reason = choice.get("finish_reason", "")
    tool_calls = choice["message"].get("tool_calls") or []

    if finish_reason == "tool_calls" and tool_calls:
        tc = tool_calls[0]
        fn_name = tc["function"]["name"]
        try:
            fn_args = json.loads(tc["function"]["arguments"])
        except json.JSONDecodeError:
            fn_args = {}
        print(f"   Tool called: {fn_name}({fn_args})")
        if fn_name != "read_file":
            failures.append(f"Tool calling: expected 'read_file', got '{fn_name}'")
        elif "path" not in fn_args:
            failures.append("Tool calling: 'path' argument missing")
        else:
            print("   OK")
    else:
        # Model may have responded in text instead of calling a tool — not necessarily wrong
        content = choice["message"].get("content", "")
        print(f"   No tool call (finish_reason={finish_reason!r}). Content: {content[:120]!r}")
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
