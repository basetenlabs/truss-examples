"""Small functional checks for a deployed Ling endpoint; not a load benchmark.

Reads BASETEN_API_KEY from the environment. Never prints or writes the key.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

MODEL = "inclusionAI/Ling-3.1-flash"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def parse_stream(lines):
    content, events, done = [], 0, False
    for raw in lines:
        line = raw.decode("utf-8").strip() if isinstance(raw, bytes) else raw.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            done = True
            break
        event = json.loads(data)
        require("error" not in event, "Streaming response contained an error")
        events += 1
        for choice in event.get("choices", []):
            text = choice.get("delta", {}).get("content")
            if text:
                content.append(text)
    require(events > 0 and done, "Missing SSE events or terminal [DONE]")
    return "".join(content), events


class Client:
    def __init__(self, model_id, deployment_id, api_key):
        require(re.fullmatch(r"[a-z0-9]+", model_id), "Invalid model ID")
        require(re.fullmatch(r"[a-z0-9]+", deployment_id), "Invalid deployment ID")
        require(bool(api_key), "Set BASETEN_API_KEY in the environment")
        self.url = f"https://model-{model_id}.api.baseten.co/deployment/{deployment_id}/sync/v1/chat/completions"
        self.key = api_key

    def complete(self, messages, *, thinking=False, stream=False, tools=None, max_tokens=512, return_usage=False):
        payload = {
            "model": MODEL,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stream": stream,
            "chat_template_kwargs": {"enable_thinking": thinking},
        }
        if tools is not None:
            payload.update(tools=tools, tool_choice="auto", parallel_tool_calls=False)
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode(),
            headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=1200 if return_usage else 120) as response:
                if stream:
                    return parse_stream(response)
                data = json.load(response)
        except urllib.error.HTTPError as exc:
            # Never include request headers/key in diagnostic output.
            raise RuntimeError(f"Chat request returned HTTP {exc.code}: {exc.read(2000).decode(errors='replace')}") from None
        require(data.get("choices"), "Response has no choices")
        require(data["choices"][0].get("finish_reason") != "length", "Functional check hit its output limit")
        message = data["choices"][0]["message"]
        return (message, data.get("usage", {})) if return_usage else message


def check_thinking_off(client):
    msg = client.complete([{"role": "user", "content": "What is 6 multiplied by 7? Return the numeric answer only."}])
    content = msg.get("content") or ""
    require(re.search(r"\b42\b", content), "Non-thinking arithmetic response was not 42")
    require(not msg.get("reasoning") and not msg.get("reasoning_content"), "Thinking-off response exposed a reasoning trace")
    require("<think>" not in content, "Raw thinking markers leaked into content")
    return {"expected_answer": 42, "reasoning_present": False}


def check_thinking_on(client):
    msg = client.complete([{"role": "user", "content": "Calculate 17 times 19. Think it through, then give the numeric result."}], thinking=True)
    content = msg.get("content") or ""
    trace = msg.get("reasoning") or msg.get("reasoning_content")
    require(re.search(r"\b323\b", content), "Thinking-on final answer was not 323")
    require(bool(trace), "Thinking-on response has no separated reasoning trace")
    require("<think>" not in content, "Raw thinking markers leaked into final content")
    return {"expected_answer": 323, "reasoning_present": True}


def check_streaming(client):
    content, events = client.complete([{"role": "user", "content": "Reply with the single word READY."}], stream=True)
    require("READY" in content.upper(), "Stream did not return the requested content")
    return {"sse_events": events, "terminal_done": True}


def check_tool_round_trip(client):
    tools = [{"type": "function", "function": {
        "name": "get_temperature", "description": "Read the current temperature for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"], "additionalProperties": False},
    }}]
    messages = [
        {"role": "system", "content": "Use the provided tool for current readings. Report temperatures using digits."},
        {"role": "user", "content": "What is the current temperature in Paris? Call get_temperature; do not guess."},
    ]
    msg = client.complete(messages, tools=tools)
    calls = msg.get("tool_calls") or []
    require(len(calls) == 1, "Expected exactly one parsed automatic tool call")
    call = calls[0]
    require(call.get("id"), "Tool call is missing its round-trip ID")
    require(call.get("type") == "function", "Unexpected tool call type")
    fn = call["function"]
    require(fn.get("name") == "get_temperature", "Wrong tool selected")
    args = json.loads(fn["arguments"]) if isinstance(fn.get("arguments"), str) else fn.get("arguments")
    require(isinstance(args, dict) and "paris" in str(args.get("city", "")).lower(), "Incorrect tool arguments")
    # Mock tool result, not a real weather claim or external tool execution.
    messages += [
        {"role": "assistant", "content": msg.get("content"), "tool_calls": calls},
        {"role": "tool", "tool_call_id": call["id"], "content": json.dumps({"city": "Paris", "celsius": 21})},
    ]
    final = client.complete(messages, tools=tools)
    require(not final.get("tool_calls"), "Tool result was not consumed into a final answer")
    require(re.search(r"\b21\b", final.get("content") or ""), "Final answer did not use the mock tool result")
    return {"automatic_tool": "get_temperature", "arguments_valid": True, "round_trip_valid": True}


def check_long_context(client):
    prompt = (
        "This is a long-input capacity probe. The repeated text below is filler.\n"
        + " x" * 260000
        + "\nAfter the filler: what is 6 times 7? Return the numeric answer only."
    )
    msg, usage = client.complete(
        [{"role": "user", "content": prompt}], max_tokens=32, return_usage=True
    )
    tokens = usage.get("prompt_tokens", 0)
    require(250000 <= tokens <= 262144, "Long-input token count was outside the intended near-limit window")
    require(re.search(r"\b42\b", msg.get("content") or ""), "Long-input probe did not finish with the expected answer")
    return {"prompt_tokens": tokens, "requested_output_tokens": 32, "capacity_probe_only": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--deployment-id", required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--long-context", action="store_true", help="Also send one near-limit (~260K-token) input; not a load or retrieval-accuracy test")
    args = parser.parse_args()
    client = Client(args.model_id, args.deployment_id, os.environ.get("BASETEN_API_KEY"))
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "checks": [], "passed": True}
    checks = [check_thinking_off, check_thinking_on, check_streaming, check_tool_round_trip]
    if args.long_context:
        checks.append(check_long_context)
    for check in checks:
        started = time.monotonic()
        row = {"name": check.__name__}
        try:
            row.update(passed=True, details=check(client))
        except Exception as exc:
            row.update(passed=False, error=str(exc))
            report["passed"] = False
        row["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report["checks"].append(row)
        print(f"{'PASS' if row['passed'] else 'FAIL'} {row['name']}", flush=True)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
