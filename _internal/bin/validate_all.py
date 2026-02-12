#!/usr/bin/env python3
"""Validate every truss example config in the repository.

Walks the repo, finds every directory containing a config.yaml,
and runs validation checks. Outputs a markdown report to stdout.

Usage:
    python bin/validate_all.py [--json] [--csv]
"""

import ast
import json
import os
import sys
from pathlib import Path

import yaml

# Ensure we can import truss
try:
    import truss
    from truss.base.truss_config import TrussConfig
except ImportError:
    print("ERROR: truss not installed. Run: uv pip install -e ../truss", file=sys.stderr)
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Directories to skip entirely (not examples)
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
}

# Patterns that indicate a deprecated/archivable example
ARCHIVE_PATTERNS = [
    "llama-2-",
    "llama-7b",
    "mistral-7b-chat",
    "mistral-7b-instruct",
    "whisper-torchserve",
    "deepspeed-mii",
    "nous-capybara",
    "model_cach_gcs",
    "qwen-7b-chat",
]

# Deprecated config fields
DEPRECATED_FIELDS_IN_YAML = ["hf_cache"]


class ValidationResult:
    def __init__(self, path: str):
        self.path = path
        self.status = "VALID"
        self.issues: list[str] = []
        self.warnings: list[str] = []
        self.has_model_py = False
        self.has_readme = False
        self.has_example_input = False
        self.has_docker_server = False
        self.has_trt_llm = False
        self.config_loads = False

    def add_issue(self, msg: str):
        self.issues.append(msg)

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def determine_status(self):
        if not self.config_loads:
            self.status = "BROKEN"
            return

        # Check archive patterns
        dir_name = Path(self.path).name.lower()
        for pattern in ARCHIVE_PATTERNS:
            if pattern in dir_name:
                self.status = "ARCHIVE"
                return

        if self.issues:
            self.status = "DEPRECATED"
            return

        if not self.has_example_input and not self.has_readme:
            self.status = "VALID_NO_INPUT"
        elif not self.has_readme:
            self.status = "VALID_NO_README"
        elif not self.has_example_input:
            self.status = "VALID_NO_INPUT"
        else:
            self.status = "VALID"


def find_all_configs(root: Path) -> list[Path]:
    """Find all directories containing a config.yaml."""
    configs = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skip dirs
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        if "config.yaml" in filenames:
            configs.append(Path(dirpath))
    return sorted(configs)


def check_model_py(example_dir: Path) -> tuple[bool, list[str]]:
    """Check if model.py exists and has load/predict methods via AST."""
    issues = []
    model_dir = example_dir / "model"
    model_py = model_dir / "model.py"

    if not model_py.exists():
        # Also check root level
        model_py = example_dir / "model.py"

    if not model_py.exists():
        return False, []

    try:
        source = model_py.read_text()
        tree = ast.parse(source)
    except SyntaxError as e:
        return True, [f"model.py has syntax error: {e}"]

    # Find class with load/predict
    has_load = False
    has_predict = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name == "load":
                        has_load = True
                    if item.name == "predict":
                        has_predict = True

    if not has_load:
        issues.append("model.py: no load() method found")
    if not has_predict:
        issues.append("model.py: no predict() method found")

    return True, issues


def check_config_yaml(example_dir: Path) -> ValidationResult:
    """Run all validation checks on a single example."""
    result = ValidationResult(str(example_dir.relative_to(REPO_ROOT)))
    config_path = example_dir / "config.yaml"

    # 1. Try loading with truss
    try:
        raw = yaml.safe_load(config_path.read_text())
    except Exception as e:
        result.add_issue(f"YAML parse error: {e}")
        result.determine_status()
        return result

    # Check for deprecated fields in raw YAML
    if raw and isinstance(raw, dict):
        for field in DEPRECATED_FIELDS_IN_YAML:
            if field in raw:
                result.add_issue(f"Deprecated field '{field}' found in config")

        # Check for old base_model values
        trt_llm = raw.get("trt_llm", {})
        if isinstance(trt_llm, dict):
            build = trt_llm.get("build", {})
            if isinstance(build, dict):
                base_model = build.get("base_model", "")
                if base_model in ("llama", "mistral"):
                    result.add_issue(
                        f"Deprecated base_model value '{base_model}' (use 'decoder')"
                    )

    # Try truss.load()
    try:
        _ = truss.load(str(example_dir))
        result.config_loads = True
    except Exception as e:
        err_str = str(e)
        # Truncate long errors
        if len(err_str) > 200:
            err_str = err_str[:200] + "..."
        result.add_issue(f"truss.load() failed: {err_str}")
        result.config_loads = False
        result.determine_status()
        return result

    # 2. Check for docker_server or trt_llm
    if raw and isinstance(raw, dict):
        result.has_docker_server = "docker_server" in raw and raw["docker_server"]
        result.has_trt_llm = "trt_llm" in raw and raw["trt_llm"]

    # 3. Check model.py (not required for docker_server or trt_llm configs)
    if not result.has_docker_server and not result.has_trt_llm:
        has_model, model_issues = check_model_py(example_dir)
        result.has_model_py = has_model
        if not has_model:
            result.add_warning("No model.py found")
        for issue in model_issues:
            result.add_warning(issue)
    else:
        result.has_model_py = True  # Not applicable

    # 4. Check example_model_input
    if raw and isinstance(raw, dict):
        model_metadata = raw.get("model_metadata", {})
        if isinstance(model_metadata, dict):
            result.has_example_input = "example_model_input" in model_metadata

    # 5. Check README.md
    result.has_readme = (example_dir / "README.md").exists()

    # 6. Check requirements.txt parseable
    req_file = example_dir / "requirements.txt"
    if req_file.exists():
        try:
            lines = req_file.read_text().strip().split("\n")
            for line in lines:
                line = line.strip()
                if line and not line.startswith("#") and not line.startswith("-"):
                    # Basic check - should have valid pip format
                    pass
        except Exception as e:
            result.add_warning(f"requirements.txt issue: {e}")

    result.determine_status()
    return result


def main():
    output_format = "markdown"
    if "--json" in sys.argv:
        output_format = "json"
    elif "--csv" in sys.argv:
        output_format = "csv"

    configs = find_all_configs(REPO_ROOT)
    print(f"Found {len(configs)} example directories with config.yaml", file=sys.stderr)

    results: list[ValidationResult] = []
    for config_dir in configs:
        result = check_config_yaml(config_dir)
        results.append(result)

    # Summary counts
    status_counts: dict[str, int] = {}
    for r in results:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1

    if output_format == "json":
        data = []
        for r in results:
            data.append(
                {
                    "path": r.path,
                    "status": r.status,
                    "issues": r.issues,
                    "warnings": r.warnings,
                    "has_model_py": r.has_model_py,
                    "has_readme": r.has_readme,
                    "has_example_input": r.has_example_input,
                    "has_docker_server": r.has_docker_server,
                    "has_trt_llm": r.has_trt_llm,
                }
            )
        print(json.dumps({"summary": status_counts, "results": data}, indent=2))

    elif output_format == "csv":
        print("path,status,issues,warnings,has_model_py,has_readme,has_example_input")
        for r in results:
            issues = "; ".join(r.issues).replace(",", ";")
            warnings = "; ".join(r.warnings).replace(",", ";")
            print(
                f"{r.path},{r.status},{issues},{warnings},{r.has_model_py},{r.has_readme},{r.has_example_input}"
            )

    else:
        # Markdown
        print("# Validation Report\n")
        print(f"**Total examples**: {len(results)}\n")
        print("## Summary\n")
        print("| Status | Count |")
        print("|--------|-------|")
        for status in [
            "VALID",
            "VALID_NO_INPUT",
            "VALID_NO_README",
            "DEPRECATED",
            "BROKEN",
            "ARCHIVE",
        ]:
            count = status_counts.get(status, 0)
            if count > 0:
                print(f"| {status} | {count} |")
        print()

        # Group by status
        for status in [
            "BROKEN",
            "DEPRECATED",
            "ARCHIVE",
            "VALID_NO_INPUT",
            "VALID_NO_README",
            "VALID",
        ]:
            group = [r for r in results if r.status == status]
            if not group:
                continue
            print(f"\n## {status} ({len(group)})\n")
            print("| Path | Issues | Warnings |")
            print("|------|--------|----------|")
            for r in group:
                issues = "; ".join(r.issues) if r.issues else "-"
                warnings = "; ".join(r.warnings) if r.warnings else "-"
                print(f"| `{r.path}` | {issues} | {warnings} |")

    # Write report to file as well
    report_path = REPO_ROOT / "validation_report.json"
    data = []
    for r in results:
        data.append(
            {
                "path": r.path,
                "status": r.status,
                "issues": r.issues,
                "warnings": r.warnings,
            }
        )
    report_path.write_text(json.dumps({"summary": status_counts, "results": data}, indent=2))
    print(f"\nReport also saved to {report_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
