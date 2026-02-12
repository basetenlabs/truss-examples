#!/usr/bin/env python3
"""Generate standardized README.md for every non-archived example directory."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SKIP_DIRS = {"_archive", "_internal", ".git", ".github", ".venv", "__pycache__"}


# ---------------------------------------------------------------------------
# Metadata extraction helpers
# ---------------------------------------------------------------------------


def extract_hf_id(config: dict) -> str | None:
    # 1. model_metadata.repo_id
    repo_id = config.get("model_metadata", {}).get("repo_id")
    if repo_id:
        return repo_id

    # 2. trt_llm checkpoint repo
    trt = config.get("trt_llm", {})
    repo = trt.get("build", {}).get("checkpoint_repository", {}).get("repo")
    if repo:
        return repo

    # 3. model_cache first entry
    caches = config.get("model_cache", [])
    if caches and isinstance(caches, list):
        first = caches[0] if caches else {}
        rid = first.get("repo_id")
        if rid:
            return rid

    # 4. Parse from docker_server.start_command
    cmd = config.get("docker_server", {}).get("start_command", "")
    for token in cmd.split():
        if "/" in token and not token.startswith("-") and not token.startswith("/"):
            # Looks like org/model — strip any trailing punctuation
            cleaned = token.strip("\"'")
            if re.match(r"^[\w.-]+/[\w.-]+", cleaned):
                return cleaned

    return None


def detect_engine(config: dict) -> str:
    base_image = str(
        config.get("base_image", {}).get("image", "")
        if isinstance(config.get("base_image"), dict)
        else ""
    )
    start_cmd = config.get("docker_server", {}).get("start_command", "")
    requirements = [str(r) for r in config.get("requirements", [])]
    req_str = " ".join(requirements)

    if config.get("trt_llm"):
        build = config["trt_llm"].get("build", {})
        base_model = build.get("base_model", "")
        if base_model in ("encoder", "encoder_bert"):
            return "BEI (TensorRT)"
        return "TRT-LLM"
    if "vllm" in base_image.lower() or "vllm" in start_cmd.lower():
        return "vLLM"
    if "sglang" in start_cmd.lower() or "sglang" in req_str.lower():
        return "SGLang"
    if (
        "text-embeddings" in base_image.lower()
        or "text-embeddings-router" in start_cmd.lower()
    ):
        return "TEI (HuggingFace)"
    if config.get("docker_server"):
        return "Docker Server"
    return "Custom (Truss)"


def infer_task_type(config: dict, category: str, dir_name: str) -> str:
    tags = config.get("model_metadata", {}).get("tags", [])
    if not isinstance(tags, list):
        tags = []

    # Check tags first
    for tag in tags:
        tag_l = tag.lower()
        if "text-to-speech" in tag_l or "tts" in tag_l:
            return "Text-to-speech"
        if "speech-to-text" in tag_l or "stt" in tag_l or "transcription" in tag_l:
            return "Speech-to-text"
        if "image-generation" in tag_l:
            return "Image generation"
        if "embedding" in tag_l:
            return "Embeddings"
        if "rerank" in tag_l:
            return "Reranking"
        if "classification" in tag_l:
            return "Classification"

    # Infer from category path
    cat_l = category.lower()
    if "llm" in cat_l or "optimized" in cat_l:
        return "Text generation"
    if "embedding" in cat_l:
        # Check if it's a reranker
        if "rerank" in dir_name.lower():
            return "Reranking"
        if (
            "classification" in dir_name.lower()
            or "reward" in dir_name.lower()
            or "ner" in dir_name.lower()
        ):
            return "Classification"
        return "Embeddings"
    if "image" in cat_l:
        return "Image generation"
    if "audio" in cat_l:
        if (
            "tts" in dir_name.lower()
            or "voice" in dir_name.lower()
            or "speech" in dir_name.lower()
            or "kokoro" in dir_name.lower()
            or "chatterbox" in dir_name.lower()
            or "metavoice" in dir_name.lower()
            or "sesame" in dir_name.lower()
        ):
            return "Text-to-speech"
        if "whisper" in dir_name.lower() or "transcri" in dir_name.lower():
            return "Speech-to-text"
        if "music" in dir_name.lower() or "audiogen" in dir_name.lower():
            return "Audio generation"
        if "ultravox" in dir_name.lower():
            return "Audio understanding"
        return "Audio"
    if "infrastructure" in cat_l:
        return "Infrastructure / Custom server"
    if "tutorial" in cat_l:
        return "Tutorial"
    return "ML inference"


def extract_quantization(config: dict, dir_name: str) -> str | None:
    # From config
    trt = config.get("trt_llm", {})
    qt = trt.get("build", {}).get("quantization_type")
    if qt:
        return qt.upper().replace("_", " ")

    # From directory name
    name_l = dir_name.lower()
    for pat, label in [
        ("fp4", "FP4"),
        ("fp8", "FP8"),
        ("int8", "INT8"),
        ("int4", "INT4"),
        ("awq", "AWQ"),
        ("gptq", "GPTQ"),
        ("bnb", "BnB 4-bit"),
    ]:
        if pat in name_l:
            return label
    return None


def extract_gpu(config: dict) -> str:
    acc = config.get("resources", {}).get("accelerator")
    if acc:
        return str(acc)
    if config.get("resources", {}).get("use_gpu"):
        return "GPU (unspecified)"
    return "CPU"


def requires_hf_token(config: dict) -> bool:
    secrets = config.get("secrets", {})
    if isinstance(secrets, dict) and "hf_access_token" in secrets:
        return True
    env = config.get("environment_variables", {})
    if isinstance(env, dict) and "hf_access_token" in env:
        return True
    return False


def get_api_endpoint(config: dict) -> str:
    ep = config.get("docker_server", {}).get("predict_endpoint")
    if ep:
        return ep
    rt = config.get("trt_llm", {}).get("runtime", {})
    route = rt.get("webserver_default_route")
    if route:
        return route
    # OpenAI-compatible TRT-LLM defaults
    tags = config.get("model_metadata", {}).get("tags", [])
    if isinstance(tags, list) and "openai-compatible" in tags:
        return "/v1/chat/completions"
    return "/predict"


def is_openai_compatible(config: dict) -> bool:
    tags = config.get("model_metadata", {}).get("tags", [])
    if not isinstance(tags, list):
        return False
    return "openai-compatible" in tags


def get_category(example_dir: Path) -> str:
    """Return the top-level category (llm, embeddings, image, audio, etc.)."""
    rel = example_dir.relative_to(REPO_ROOT)
    parts = rel.parts
    if len(parts) >= 1:
        return parts[0]
    return "unknown"


def get_subcategory(example_dir: Path) -> str:
    """Return deeper subcategory path for richer context."""
    rel = example_dir.relative_to(REPO_ROOT)
    parts = rel.parts
    if len(parts) >= 2:
        return "/".join(parts[:2])
    return parts[0] if parts else "unknown"


# ---------------------------------------------------------------------------
# Config highlights
# ---------------------------------------------------------------------------


def build_config_highlights(config: dict, engine: str) -> list[str]:
    highlights = []

    # Quantization
    trt = config.get("trt_llm", {})
    build = trt.get("build", {})
    qt = build.get("quantization_type")
    if qt:
        highlights.append(f"Quantization: **{qt}**")

    # Tensor parallelism
    tp = build.get("tensor_parallel_count")
    if tp and tp > 1:
        highlights.append(f"Tensor parallelism: **{tp}** GPUs")

    # Speculative decoding
    spec = build.get("speculator", {})
    if spec:
        mode = spec.get("speculative_decoding_mode", "enabled")
        highlights.append(f"Speculative decoding: **{mode}**")

    # Max sequence length
    max_seq = build.get("max_seq_len")
    if max_seq:
        highlights.append(f"Max sequence length: **{max_seq:,}**")

    # Chunked context
    if trt.get("runtime", {}).get("enable_chunked_context"):
        highlights.append("Chunked context: **enabled**")

    # Batch scheduler policy
    bsp = trt.get("runtime", {}).get("batch_scheduler_policy")
    if bsp:
        highlights.append(f"Batch scheduler policy: **{bsp}**")

    # Plugin configuration
    plugins = build.get("plugin_configuration", {})
    if plugins:
        for k, v in plugins.items():
            if v:
                highlights.append(f"Plugin: **{k}**")

    # Custom base image
    base_image = config.get("base_image", {})
    if isinstance(base_image, dict) and base_image.get("image"):
        highlights.append(f"Base image: `{base_image['image']}`")

    # Model cache / volume mounting
    caches = config.get("model_cache", [])
    if caches and isinstance(caches, list):
        for c in caches:
            if c.get("use_volume"):
                highlights.append(
                    "Model cache: **volume-mounted** for fast cold starts"
                )
                break

    # Concurrency
    conc = config.get("runtime", {}).get("predict_concurrency")
    if conc:
        highlights.append(f"Predict concurrency: **{conc}**")

    # System packages
    sys_pkgs = config.get("system_packages", [])
    if sys_pkgs:
        highlights.append(f"System packages: `{', '.join(sys_pkgs)}`")

    # Streaming via example input
    example = config.get("model_metadata", {}).get("example_model_input", {})
    if isinstance(example, dict) and example.get("stream"):
        highlights.append("Streaming: **enabled**")

    # Environment variables (non-secret)
    envs = config.get("environment_variables", {})
    if isinstance(envs, dict):
        notable = {
            k: v for k, v in envs.items() if k != "hf_access_token" and v is not None
        }
        if notable:
            highlights.append(
                f"Environment variables: {', '.join(f'`{k}`' for k in notable)}"
            )

    if not highlights:
        highlights.append(f"Engine: **{engine}**")

    return highlights


# ---------------------------------------------------------------------------
# Invoke section
# ---------------------------------------------------------------------------


def build_invoke_section(
    config: dict,
    engine: str,
    task: str,
    endpoint: str,
    hf_id: str | None,
    openai_compat: bool,
) -> str:
    example_input = config.get("model_metadata", {}).get("example_model_input")

    # --- OpenAI-compatible LLM ---
    if openai_compat and task == "Text generation":
        model_name = hf_id or "model"
        return f"""\
This model is OpenAI-compatible. You can use the OpenAI Python client or curl.

**Python (OpenAI SDK):**

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_BASETEN_API_KEY",
    base_url="https://model-<model_id>.api.baseten.co/v1",
)

response = client.chat.completions.create(
    model="{model_name}",
    messages=[{{"role": "user", "content": "What is machine learning?"}}],
    max_tokens=512,
)

print(response.choices[0].message.content)
```

**curl:**

```sh
curl -X POST https://model-<model_id>.api.baseten.co/v1/chat/completions \\
  -H "Authorization: Api-Key YOUR_BASETEN_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{{"model": "{model_name}", "messages": [{{"role": "user", "content": "What is machine learning?"}}], "max_tokens": 512}}'
```"""

    # --- BEI / TEI embeddings ---
    if endpoint == "/v1/embeddings":
        model_name = hf_id or "model"
        return f"""\
```sh
curl -X POST https://model-<model_id>.api.baseten.co/v1/embeddings \\
  -H "Authorization: Api-Key YOUR_BASETEN_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{{"input": "What is deep learning?", "model": "{model_name}"}}'
```"""

    # --- Reranker ---
    if endpoint == "/rerank" or task == "Reranking":
        return """\
```sh
curl -X POST https://model-<model_id>.api.baseten.co/rerank \\
  -H "Authorization: Api-Key YOUR_BASETEN_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"query": "What is deep learning?", "texts": ["Deep learning is a subset of machine learning.", "The weather is nice today."], "raw_scores": true}'
```"""

    # --- Use example_model_input if available ---
    if example_input:
        if isinstance(example_input, str):
            try:
                formatted = example_input
            except Exception:
                formatted = example_input
            return f"""\
```sh
curl -X POST https://model-<model_id>.api.baseten.co{endpoint} \\
  -H "Authorization: Api-Key YOUR_BASETEN_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{formatted}'
```"""
        else:
            formatted = json.dumps(example_input, indent=2)
            return f"""\
```sh
curl -X POST https://model-<model_id>.api.baseten.co{endpoint} \\
  -H "Authorization: Api-Key YOUR_BASETEN_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{formatted}'
```"""

    # --- Fallback by task ---
    if task == "Text generation":
        return f"""\
```sh
curl -X POST https://model-<model_id>.api.baseten.co{endpoint} \\
  -H "Authorization: Api-Key YOUR_BASETEN_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{{"prompt": "What is machine learning?", "max_tokens": 512}}'
```"""

    if task == "Image generation":
        return f"""\
```sh
curl -X POST https://model-<model_id>.api.baseten.co{endpoint} \\
  -H "Authorization: Api-Key YOUR_BASETEN_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{{"prompt": "A photo of a cat in a field of sunflowers"}}'
```

> The response may contain base64-encoded image data."""

    if task in ("Text-to-speech", "Audio generation"):
        return f"""\
```sh
curl -X POST https://model-<model_id>.api.baseten.co{endpoint} \\
  -H "Authorization: Api-Key YOUR_BASETEN_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{{"text": "Hello, this is a test of text to speech."}}'
```"""

    if task == "Speech-to-text":
        return f"""\
```sh
curl -X POST https://model-<model_id>.api.baseten.co{endpoint} \\
  -H "Authorization: Api-Key YOUR_BASETEN_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{{"url": "https://example.com/audio.wav"}}'
```"""

    # Generic fallback
    return f"""\
```sh
curl -X POST https://model-<model_id>.api.baseten.co{endpoint} \\
  -H "Authorization: Api-Key YOUR_BASETEN_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{{}}'
```"""


# ---------------------------------------------------------------------------
# Description generation
# ---------------------------------------------------------------------------


def build_description(
    model_name: str,
    config: dict,
    category: str,
    engine: str,
    hf_id: str | None,
    task: str,
) -> str:
    desc = config.get("description")
    if desc:
        return desc

    hf_part = f"[{hf_id}](https://huggingface.co/{hf_id})" if hf_id else model_name
    engine_article = "an" if engine[0] in "AEIOU" else "a"

    if task == "Text generation":
        return f"Deploy {hf_part} for text generation using {engine_article} {engine} engine on Baseten."
    if task == "Embeddings":
        return f"Deploy {hf_part} for generating text embeddings using {engine_article} {engine} engine on Baseten."
    if task == "Reranking":
        return f"Deploy {hf_part} as a reranker using {engine_article} {engine} engine on Baseten."
    if task == "Classification":
        return f"Deploy {hf_part} for classification using {engine_article} {engine} engine on Baseten."
    if task == "Image generation":
        return f"Deploy {hf_part} for image generation on Baseten."
    if task == "Text-to-speech":
        return f"Deploy {hf_part} for text-to-speech on Baseten."
    if task == "Speech-to-text":
        return f"Deploy {hf_part} for speech-to-text transcription on Baseten."
    if task == "Audio generation":
        return f"Deploy {hf_part} for audio generation on Baseten."
    if task == "Audio understanding":
        return f"Deploy {hf_part} for audio understanding on Baseten."
    if task == "Tutorial":
        return f"A tutorial example showing how to deploy {model_name} on Baseten."
    if task == "Infrastructure / Custom server":
        return f"Deploy {hf_part} using a custom server configuration on Baseten."

    return f"Deploy {model_name} on Baseten using {engine_article} {engine} engine."


# ---------------------------------------------------------------------------
# README rendering
# ---------------------------------------------------------------------------


def render_readme(
    model_name: str,
    description: str,
    hf_id: str | None,
    task: str,
    engine: str,
    gpu: str,
    quantization: str | None,
    openai_compat: bool,
    hf_token: bool,
    endpoint: str,
    highlights: list[str],
    invoke_section: str,
    has_model_py: bool,
    python_version: str | None,
) -> str:
    lines = []
    lines.append(f"# {model_name}\n")
    lines.append(f"{description}\n")

    # Properties table
    lines.append("| Property | Value |")
    lines.append("|----------|-------|")
    if hf_id:
        lines.append(f"| Model | [{hf_id}](https://huggingface.co/{hf_id}) |")
    lines.append(f"| Task | {task} |")
    lines.append(f"| Engine | {engine} |")
    lines.append(f"| GPU | {gpu} |")
    if quantization:
        lines.append(f"| Quantization | {quantization} |")
    if openai_compat:
        lines.append("| OpenAI compatible | Yes |")
    if python_version:
        lines.append(f"| Python | {python_version} |")
    lines.append("")

    # Deploy
    lines.append("## Deploy\n")
    if hf_token:
        lines.append(
            "> **Note:** This model requires a HuggingFace access token. Set `hf_access_token` in your Baseten secrets before deploying.\n"
        )
    lines.append("```sh\ntruss push\n```\n")

    # Invoke
    lines.append("## Invoke\n")
    lines.append(invoke_section)
    lines.append("")

    # Config highlights
    lines.append("## Configuration highlights\n")
    for h in highlights:
        lines.append(f"- {h}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def find_example_dirs() -> list[Path]:
    """Find all directories containing config.yaml, excluding archive/internal."""
    examples = []
    for root, dirs, files in os.walk(REPO_ROOT):
        # Prune skip dirs
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        if "config.yaml" in files:
            examples.append(Path(root))
    return sorted(examples)


def process_example(example_dir: Path) -> dict:
    """Process a single example directory and return metadata + generated README."""
    config_path = example_dir / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    dir_name = example_dir.name
    category = get_category(example_dir)

    model_name = (
        config.get("model_name") or dir_name.replace("-", " ").replace("_", " ").title()
    )
    hf_id = extract_hf_id(config)
    engine = detect_engine(config)
    task = infer_task_type(config, category, dir_name)
    gpu = extract_gpu(config)
    quantization = extract_quantization(config, dir_name)
    openai_compat = is_openai_compatible(config)
    hf_token = requires_hf_token(config)
    endpoint = get_api_endpoint(config)
    python_version = config.get("python_version")
    has_model_py = (example_dir / "model" / "model.py").exists()
    highlights = build_config_highlights(config, engine)

    description = build_description(model_name, config, category, engine, hf_id, task)
    invoke = build_invoke_section(config, engine, task, endpoint, hf_id, openai_compat)

    readme = render_readme(
        model_name=model_name,
        description=description,
        hf_id=hf_id,
        task=task,
        engine=engine,
        gpu=gpu,
        quantization=quantization,
        openai_compat=openai_compat,
        hf_token=hf_token,
        endpoint=endpoint,
        highlights=highlights,
        invoke_section=invoke,
        has_model_py=has_model_py,
        python_version=python_version,
    )

    return {
        "dir": str(example_dir),
        "model_name": model_name,
        "hf_id": hf_id,
        "engine": engine,
        "task": task,
        "readme": readme,
    }


def main():
    examples = find_example_dirs()
    print(f"Found {len(examples)} example directories\n")

    generated = 0
    missing_hf = []

    for example_dir in examples:
        try:
            result = process_example(example_dir)
        except Exception as e:
            print(f"  ERROR: {example_dir.relative_to(REPO_ROOT)}: {e}")
            continue

        readme_path = example_dir / "README.md"
        readme_path.write_text(result["readme"])
        generated += 1

        rel = example_dir.relative_to(REPO_ROOT)
        status = "ok" if result["hf_id"] else "no HF ID"
        print(f"  {rel} [{result['engine']}] — {status}")

        if not result["hf_id"]:
            missing_hf.append(str(rel))

    print("\n--- Summary ---")
    print(f"Generated: {generated}/{len(examples)}")
    print(f"Missing HuggingFace ID: {len(missing_hf)}")
    if missing_hf:
        print("Directories without HF ID:")
        for d in missing_hf:
            print(f"  - {d}")


if __name__ == "__main__":
    main()
