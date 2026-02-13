#!/usr/bin/env python3
"""Auto-discover all testable example directories in the repository.

Walks the repo looking for directories containing a config.yaml,
skipping internal/archived paths and any paths listed in ci_excludes.yaml.

Outputs a sorted JSON array of relative paths to stdout.
Used by the GitHub Actions workflow to generate the CI matrix.

Usage:
    python _internal/bin/discover_examples.py           # compact JSON
    python _internal/bin/discover_examples.py --pretty   # indented JSON
"""

import json
import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

SKIP_DIRS = {".git", ".github", "__pycache__", "node_modules", ".venv", "venv"}

SKIP_PREFIXES = ("_archive", "_internal")


def load_excludes(repo_root: Path) -> set[str]:
    """Load excluded paths from ci_excludes.yaml if it exists."""
    excludes_path = repo_root / "ci_excludes.yaml"
    if not excludes_path.exists():
        return set()
    with open(excludes_path) as f:
        data = yaml.safe_load(f)
    if not data or not isinstance(data, dict):
        return set()
    return set(data.get("exclude", []) or [])


def discover_examples(repo_root: Path) -> list[str]:
    """Find all directories containing a config.yaml, minus excludes."""
    excludes = load_excludes(repo_root)
    examples = []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        if "config.yaml" in filenames:
            rel = str(Path(dirpath).relative_to(repo_root))
            if any(rel.startswith(prefix) for prefix in SKIP_PREFIXES):
                continue
            if rel in excludes:
                continue
            examples.append(rel)
    return sorted(examples)


def main():
    pretty = "--pretty" in sys.argv
    examples = discover_examples(REPO_ROOT)
    indent = 2 if pretty else None
    print(json.dumps(examples, indent=indent))


if __name__ == "__main__":
    main()
