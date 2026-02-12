#!/usr/bin/env python3
"""Comprehensive test suite for truss-examples repository.

Runs all validation that can be done locally without deploying models:
  1. Config validation (YAML parsing, truss.load())
  2. README existence and structure
  3. README ↔ config consistency (endpoints, secrets, model names)
  4. Directory naming conventions
  5. Link/path validation (README links point to real dirs)
  6. example_model_input format validation
  7. Requirements pinning check
  8. CI path validation

Usage:
    python _internal/bin/test_all.py
    python _internal/bin/test_all.py --verbose
    python _internal/bin/test_all.py --category llm
"""

import os
import re
import sys
from pathlib import Path

import yaml

try:
    import truss
except ImportError:
    print(
        "ERROR: truss not installed. Run: uv pip install -e ../truss", file=sys.stderr
    )
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VERBOSE = "--verbose" in sys.argv or "-v" in sys.argv
CATEGORY_FILTER = None
for i, arg in enumerate(sys.argv):
    if arg == "--category" and i + 1 < len(sys.argv):
        CATEGORY_FILTER = sys.argv[i + 1]

SKIP_DIRS = {
    ".git",
    ".github",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "templating",
    "assets",
    "bin",
    "dockerfiles",
    "packages",
    "sample_images",
    "examples",  # skip nested examples dirs for dir naming checks
}

CUSTOMER_CATEGORIES = [
    "tutorials",
    "llm",
    "embeddings",
    "image",
    "audio",
    "optimized",
    "infrastructure",
]

# Counters
passed = 0
failed = 0
warnings = 0
failures = []


def log_pass(msg):
    global passed
    passed += 1
    if VERBOSE:
        print(f"  PASS  {msg}")


def log_fail(msg):
    global failed
    failed += 1
    failures.append(msg)
    print(f"  FAIL  {msg}")


def log_warn(msg):
    global warnings
    warnings += 1
    if VERBOSE:
        print(f"  WARN  {msg}")


def find_all_configs(root: Path) -> list[Path]:
    """Find all directories containing a config.yaml."""
    configs = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d
            for d in dirnames
            if d
            not in {".git", ".github", "__pycache__", "node_modules", ".venv", "venv"}
        ]
        if "config.yaml" in filenames:
            rel = Path(dirpath).relative_to(root)
            # Skip _archive
            if str(rel).startswith("_archive"):
                continue
            if CATEGORY_FILTER and not str(rel).startswith(CATEGORY_FILTER):
                continue
            configs.append(Path(dirpath))
    return sorted(configs)


# ─── Test 1: Config validation ───────────────────────────────────────────────


def test_config_validation():
    print("\n== Test 1: Config Validation (YAML + truss.load) ==")
    configs = find_all_configs(REPO_ROOT)
    print(f"   Found {len(configs)} configs to validate")

    broken = []
    for config_dir in configs:
        rel = config_dir.relative_to(REPO_ROOT)
        config_path = config_dir / "config.yaml"
        try:
            raw = yaml.safe_load(config_path.read_text())
        except Exception as e:
            log_fail(f"{rel}: YAML parse error: {e}")
            broken.append(str(rel))
            continue

        try:
            truss.load(str(config_dir))
            log_pass(f"{rel}: config loads")
        except Exception as e:
            err = str(e)[:150]
            # Allow known internal/archived breakages
            if str(rel).startswith("_internal") or "/_archive/" in str(rel):
                log_warn(f"{rel}: truss.load() failed (internal/archived): {err}")
            else:
                log_fail(f"{rel}: truss.load() failed: {err}")
                broken.append(str(rel))

    return broken


# ─── Test 2: README existence ────────────────────────────────────────────────


def test_readme_existence():
    print("\n== Test 2: README Existence ==")
    configs = find_all_configs(REPO_ROOT)
    missing = []

    for config_dir in configs:
        rel = config_dir.relative_to(REPO_ROOT)
        if (
            str(rel).startswith("_internal")
            or "/_archive/" in str(rel)
            or str(rel).startswith("_archive")
        ):
            continue
        # Skip _archive at any level
        if "_archive" in rel.parts:
            continue
        readme = config_dir / "README.md"
        if readme.exists():
            log_pass(f"{rel}: has README.md")
        else:
            log_fail(f"{rel}: missing README.md")
            missing.append(str(rel))

    # Check category READMEs
    for cat in CUSTOMER_CATEGORIES:
        cat_readme = REPO_ROOT / cat / "README.md"
        if cat_readme.exists():
            log_pass(f"{cat}/README.md exists")
        else:
            log_fail(f"{cat}/README.md missing")

    return missing


# ─── Test 3: README ↔ Config consistency ─────────────────────────────────────


def test_readme_config_consistency():
    print("\n== Test 3: README ↔ Config Consistency ==")
    configs = find_all_configs(REPO_ROOT)

    for config_dir in configs:
        rel = config_dir.relative_to(REPO_ROOT)
        if str(rel).startswith("_internal"):
            continue
        readme_path = config_dir / "README.md"
        config_path = config_dir / "config.yaml"

        if not readme_path.exists():
            continue

        try:
            raw = yaml.safe_load(config_path.read_text())
        except Exception:
            continue

        if not raw or not isinstance(raw, dict):
            continue

        readme_text = readme_path.read_text()

        # Check: if config has hf_access_token secret, README should mention it
        secrets = raw.get("secrets", {})
        if secrets and isinstance(secrets, dict) and "hf_access_token" in secrets:
            if (
                "hf_access_token" in readme_text
                or "HuggingFace access token" in readme_text
            ):
                log_pass(f"{rel}: README mentions HF token (config requires it)")
            else:
                log_warn(
                    f"{rel}: config has hf_access_token but README doesn't mention it"
                )

        # Check: endpoint consistency
        docker_server = raw.get("docker_server", {})
        if isinstance(docker_server, dict):
            predict_endpoint = docker_server.get("predict_endpoint", "")
            if predict_endpoint and predict_endpoint != "/predict":
                # README should reference this endpoint
                if predict_endpoint in readme_text:
                    log_pass(f"{rel}: README uses correct endpoint {predict_endpoint}")
                elif "/predict" in readme_text and predict_endpoint not in readme_text:
                    log_warn(
                        f"{rel}: README uses /predict but config has {predict_endpoint}"
                    )

        # Check: OpenAI-compatible tag should have /v1/chat/completions in README
        tags = []
        metadata = raw.get("model_metadata", {})
        if isinstance(metadata, dict):
            tags = metadata.get("tags", [])
        if isinstance(tags, list) and "openai-compatible" in tags:
            if "/v1/chat/completions" in readme_text or "OpenAI" in readme_text:
                log_pass(f"{rel}: OpenAI-compatible model has correct invoke style")
            else:
                log_warn(
                    f"{rel}: tagged openai-compatible but README lacks /v1/chat/completions"
                )


# ─── Test 4: Directory naming conventions ────────────────────────────────────


def test_directory_naming():
    print("\n== Test 4: Directory Naming Conventions ==")

    for cat in CUSTOMER_CATEGORIES:
        cat_dir = REPO_ROOT / cat
        if not cat_dir.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(cat_dir):
            dirnames[:] = [
                d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")
            ]
            rel = Path(dirpath).relative_to(REPO_ROOT)

            # If this directory contains config.yaml, it's an example root.
            # Don't check naming of its children (model/, data/, etc. are internal).
            if "config.yaml" in filenames:
                dirnames.clear()
                continue

            for d in dirnames:
                if d.startswith("_"):
                    continue  # _archive is OK

                # Check underscores (allow go_emotions which is upstream model name)
                if "_" in d and "go_emotions" not in d:
                    log_fail(
                        f"{rel}/{d}: directory name contains underscore (use hyphens)"
                    )

                # Check for "truss" in name
                if (
                    "truss" in d.lower() and d != "truss"
                ):  # allow ngram-speculator/truss
                    log_warn(f"{rel}/{d}: directory name contains 'truss'")

                # Check PascalCase (first char uppercase suggests PascalCase)
                if d[0].isupper() and "-" in d:
                    log_warn(
                        f"{rel}/{d}: directory name uses PascalCase (prefer lowercase)"
                    )


# ─── Test 5: Link/path validation in READMEs ────────────────────────────────


def test_readme_links():
    print("\n== Test 5: README Link Validation ==")

    # Check root README
    root_readme = REPO_ROOT / "README.md"
    if root_readme.exists():
        text = root_readme.read_text()
        # Find all relative links like [text](path/)
        links = re.findall(r"\[.*?\]\(([^)]+)\)", text)
        for link in links:
            if link.startswith("http") or link.startswith("#"):
                continue
            # Strip trailing /
            clean = link.rstrip("/")
            target = REPO_ROOT / clean
            if target.exists() or (REPO_ROOT / clean.split("/")[0]).exists():
                log_pass(f"README.md: link to {clean} is valid")
            else:
                log_fail(f"README.md: broken link to {clean}")

    # Check category READMEs
    for cat in CUSTOMER_CATEGORIES:
        cat_readme = REPO_ROOT / cat / "README.md"
        if not cat_readme.exists():
            continue
        text = cat_readme.read_text()
        links = re.findall(r"\[.*?\]\(([^)]+)\)", text)
        for link in links:
            if link.startswith("http") or link.startswith("#"):
                continue
            clean = link.rstrip("/")
            target = REPO_ROOT / cat / clean
            if target.exists():
                log_pass(f"{cat}/README.md: link to {clean} is valid")
            else:
                log_fail(f"{cat}/README.md: broken link to {clean}")

    # Check CONTRIBUTING.md
    contrib = REPO_ROOT / "CONTRIBUTING.md"
    if contrib.exists():
        text = contrib.read_text()
        # Find paths in the table (backtick-wrapped)
        paths = re.findall(r"`([a-z][\w/-]+)`", text)
        for p in paths:
            if (
                "/" in p
                and not p.endswith(".py")
                and not p.endswith(".yaml")
                and not p.endswith(".yml")
            ):
                target = REPO_ROOT / p
                if target.exists():
                    log_pass(f"CONTRIBUTING.md: path {p} exists")
                elif not any(
                    p.endswith(ext) for ext in [".yaml", ".py", ".md", ".json"]
                ):
                    # Check if it exists as a subdir within any category
                    found = any(
                        (REPO_ROOT / cat / p).exists() for cat in CUSTOMER_CATEGORIES
                    )
                    if found:
                        log_pass(
                            f"CONTRIBUTING.md: path {p} exists (relative to category)"
                        )
                    else:
                        log_warn(f"CONTRIBUTING.md: path {p} may not exist")


# ─── Test 6: example_model_input validation ──────────────────────────────────


def test_example_model_input():
    print("\n== Test 6: example_model_input Validation ==")
    configs = find_all_configs(REPO_ROOT)
    missing_count = 0

    for config_dir in configs:
        rel = config_dir.relative_to(REPO_ROOT)
        if str(rel).startswith("_internal") or "_archive" in rel.parts:
            continue

        config_path = config_dir / "config.yaml"
        try:
            raw = yaml.safe_load(config_path.read_text())
        except Exception:
            continue

        if not raw or not isinstance(raw, dict):
            continue

        metadata = raw.get("model_metadata", {})
        if not isinstance(metadata, dict):
            continue

        example_input = metadata.get("example_model_input")
        if example_input is None:
            missing_count += 1
            log_warn(f"{rel}: missing example_model_input")
            continue

        # Validate input format based on category
        rel_str = str(rel)
        if rel_str.startswith("llm/"):
            # LLMs: prompt, messages, query, text, message, image_url, queries all valid
            if isinstance(example_input, dict):
                valid_keys = {
                    "prompt",
                    "messages",
                    "query",
                    "text",
                    "message",
                    "image_url",
                    "queries",
                    "model",
                }
                if valid_keys & set(example_input.keys()):
                    log_pass(f"{rel}: example_model_input has valid LLM format")
                else:
                    log_warn(
                        f"{rel}: LLM example_model_input missing prompt/messages/query"
                    )
            elif isinstance(example_input, str):
                log_pass(f"{rel}: example_model_input is a string (direct input)")

        elif rel_str.startswith("embeddings/"):
            # Embeddings: input (standard), query/texts (rerankers), text/inputs (classifiers/NER), url (CLIP)
            if isinstance(example_input, dict):
                valid_keys = {
                    "input",
                    "inputs",
                    "encoding_format",
                    "query",
                    "texts",
                    "text",
                    "model",
                    "sentences",
                    "url",
                }
                if valid_keys & set(example_input.keys()):
                    log_pass(f"{rel}: example_model_input has valid embedding format")
                else:
                    log_warn(
                        f"{rel}: embedding example_model_input may have wrong format"
                    )
            elif isinstance(example_input, str):
                log_pass(f"{rel}: example_model_input is a string (direct input)")
            else:
                log_warn(f"{rel}: embedding example_model_input may have wrong format")

        elif rel_str.startswith("image/"):
            # Image: prompt, image, instances, workflow_values (comfyui), input_image, reference_image, image_url
            if isinstance(example_input, dict):
                valid_keys = {
                    "prompt",
                    "image",
                    "instances",
                    "workflow",
                    "workflow_values",
                    "url",
                    "input_image",
                    "reference_image",
                    "bbox",
                    "image_url",
                    "text",
                }
                if valid_keys & set(example_input.keys()):
                    log_pass(f"{rel}: example_model_input has valid image format")
                else:
                    log_warn(f"{rel}: image example_model_input may have wrong format")
            else:
                log_warn(f"{rel}: image example_model_input may have wrong format")

        elif rel_str.startswith("audio/"):
            if isinstance(example_input, dict):
                log_pass(f"{rel}: example_model_input is a dict")
            elif isinstance(example_input, str):
                log_pass(f"{rel}: example_model_input is a string")

        else:
            log_pass(f"{rel}: has example_model_input")

    if missing_count:
        print(f"   {missing_count} configs still missing example_model_input")


# ─── Test 7: Requirements pinning ────────────────────────────────────────────


def test_requirements_pinning():
    print("\n== Test 7: Requirements Pinning ==")
    configs = find_all_configs(REPO_ROOT)
    unpinned_count = 0

    for config_dir in configs:
        rel = config_dir.relative_to(REPO_ROOT)
        if str(rel).startswith("_internal") or str(rel).startswith("_archive"):
            continue

        config_path = config_dir / "config.yaml"
        try:
            raw = yaml.safe_load(config_path.read_text())
        except Exception:
            continue

        if not raw or not isinstance(raw, dict):
            continue

        requirements = raw.get("requirements", [])
        if not isinstance(requirements, list):
            continue

        # Check if using requirements_file instead
        if raw.get("requirements_file"):
            continue

        unpinned = []
        for req in requirements:
            if not isinstance(req, str):
                continue
            req = req.strip()
            if not req or req.startswith("#") or req.startswith("-"):
                continue
            if req.startswith("git+"):
                # git+ URLs should pin to a commit hash, not @master or @main
                if "@master" in req or "@main" in req:
                    unpinned.append(req)
                continue
            # Check for version pin
            if (
                "==" not in req
                and ">=" not in req
                and "<=" not in req
                and "~=" not in req
            ):
                unpinned.append(req)

        if unpinned:
            unpinned_count += 1
            for u in unpinned:
                log_warn(f"{rel}: unpinned requirement: {u}")
        elif requirements:
            log_pass(f"{rel}: all requirements pinned")

    if unpinned_count:
        print(f"   {unpinned_count} configs have unpinned requirements")


# ─── Test 8: CI path validation ──────────────────────────────────────────────


def test_ci_paths():
    print("\n== Test 8: CI Path Validation ==")
    ci_path = REPO_ROOT / "ci.yaml"
    if not ci_path.exists():
        log_fail("ci.yaml not found")
        return

    with open(ci_path) as f:
        ci = yaml.safe_load(f)

    tests = ci.get("tests", [])
    print(f"   {len(tests)} CI test paths")

    for test_path in tests:
        full_path = REPO_ROOT / test_path
        if full_path.exists():
            # Also verify it loads
            try:
                truss.load(str(full_path))
                log_pass(f"ci.yaml: {test_path} exists and loads")
            except Exception as e:
                log_fail(
                    f"ci.yaml: {test_path} exists but truss.load() fails: {str(e)[:100]}"
                )
        else:
            log_fail(f"ci.yaml: {test_path} does not exist")


# ─── Test 9: TRT-LLM openai-compatible tag ──────────────────────────────────


def test_trt_llm_tags():
    print("\n== Test 9: TRT-LLM openai-compatible Tag ==")
    configs = find_all_configs(REPO_ROOT)
    missing = 0

    for config_dir in configs:
        rel = config_dir.relative_to(REPO_ROOT)
        if "_archive" in rel.parts or "_internal" in rel.parts:
            continue

        config_path = config_dir / "config.yaml"
        try:
            raw = yaml.safe_load(config_path.read_text())
        except Exception:
            continue

        if not raw or not isinstance(raw, dict):
            continue

        if not raw.get("trt_llm"):
            continue

        tags = raw.get("model_metadata", {}).get("tags", [])
        if not isinstance(tags, list):
            tags = []

        has_oai = "openai-compatible" in tags
        has_legacy = "force-legacy-api-non-openai-compatible" in tags
        if has_oai or has_legacy:
            log_pass(f"{rel}: TRT-LLM has API compatibility tag")
        else:
            log_fail(f"{rel}: TRT-LLM missing openai-compatible or force-legacy tag")
            missing += 1

    if missing:
        print(f"   {missing} TRT-LLM configs missing API compatibility tag")


# ─── Test 10: model.py syntax validation ────────────────────────────────────


def test_model_py_syntax():
    print("\n== Test 10: model.py Syntax Validation ==")
    import ast

    model_files = sorted(REPO_ROOT.glob("**/model/model.py"))
    checked = 0

    for mf in model_files:
        rel = mf.relative_to(REPO_ROOT)
        if "_archive" in rel.parts or "_internal" in rel.parts:
            continue
        checked += 1
        try:
            ast.parse(mf.read_text())
            log_pass(f"{rel}: syntax OK")
        except SyntaxError as e:
            log_fail(f"{rel}: syntax error: {e}")

    print(f"   Checked {checked} model.py files")


# ─── Test 10: CI completeness ───────────────────────────────────────────────


def test_ci_completeness():
    print("\n== Test 11: CI Completeness (ci.yaml covers all examples) ==")
    ci_path = REPO_ROOT / "ci.yaml"
    if not ci_path.exists():
        log_fail("ci.yaml not found")
        return

    with open(ci_path) as f:
        ci = yaml.safe_load(f)

    ci_paths = set(ci.get("tests", []))
    configs = find_all_configs(REPO_ROOT)
    missing_from_ci = []

    for config_dir in configs:
        rel = config_dir.relative_to(REPO_ROOT)
        rel_str = str(rel)
        if rel_str.startswith("_internal") or "_archive" in rel.parts:
            continue
        if rel_str not in ci_paths:
            missing_from_ci.append(rel_str)
            log_fail(f"ci.yaml missing: {rel_str}")
        else:
            log_pass(f"ci.yaml covers: {rel_str}")

    if missing_from_ci:
        print(f"   {len(missing_from_ci)} examples not in ci.yaml")
    else:
        print(f"   ci.yaml covers all {len(ci_paths)} examples")


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("  Truss Examples - Comprehensive Test Suite")
    print("=" * 60)

    test_config_validation()
    test_readme_existence()
    test_readme_config_consistency()
    test_directory_naming()
    test_readme_links()
    test_example_model_input()
    test_requirements_pinning()
    test_ci_paths()
    test_trt_llm_tags()
    test_model_py_syntax()
    test_ci_completeness()

    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed, {warnings} warnings")
    print("=" * 60)

    if failures:
        print(f"\n  {len(failures)} failure(s):")
        for f in failures:
            print(f"    - {f}")
        print()
        sys.exit(1)
    else:
        print("\n  All tests passed!\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
