"""Exec sglang.launch_server with this preset's default chat template kwargs.

--default-chat-template-kwargs takes a JSON object. Passing it here as one argv
element avoids the shell-quoting loss described in llm/AGENT.md. The defaults
live in chat_template_defaults.json next to this file. Per-request
chat_template_kwargs still take precedence (SGLang v0.5.21 fields/serving.py).
"""

import json
import os
import pathlib
import sys

DEFAULTS = json.loads(pathlib.Path(__file__).with_name("chat_template_defaults.json").read_text())

if __name__ == "__main__":
    args = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        *sys.argv[1:],
        "--default-chat-template-kwargs",
        json.dumps(DEFAULTS),
    ]
    os.execv(sys.executable, args)
