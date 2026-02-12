# _internal/

Development tooling, CI infrastructure, and codegen scripts for the truss-examples repository. Nothing in this directory is a user-facing example.

## Directory layout

```
_internal/
├── bin/                        # CLI scripts for testing, validation, and README generation
├── templates/                  # Legacy TRT-LLM / Whisper codegen templates
├── templating/                 # Programmatic config generation for BEI, TEI, Briton, BISV2
├── baseten-inference-stack-v2-templates/  # BIS V2 model configs (DeepSeek, Llama 4, Qwen, etc.)
├── trt-llm-engine-builder-templates/     # TRT-LLM engine builder configs (Llama 3.1 variants)
├── assets/                     # Static assets (ComfyUI screenshots, workflow JSON)
├── dockerfiles/                # Custom Dockerfiles (ComfyUI)
├── essential/                  # Config for Essential AI model (vLLM-based)
├── internal/                   # Config for internal Briton speculative decoding test
└── validation_report.json      # Output from validate_all.py
```

## Scripts

### bin/

| Script | Description |
|--------|-------------|
| `test_all.py` | Comprehensive local test suite: config validation, README checks, naming conventions, link validation, CI completeness |
| `validate_all.py` | Walks every config.yaml and produces a structured validation report (markdown, JSON, or CSV) |
| `validate_ci.py` | Loads every path listed in the top-level `ci.yaml` with `truss.load()` to verify CI paths are valid |
| `test_example.py` | CI script: detects changed model from git diff, pushes to staging, runs inference, cleans up old deployments |
| `test_truss_deploy.py` | Deploys a truss to staging via `truss push`, invokes inference with `example_model_input`, deactivates after |
| `generate_readmes.py` | Auto-generates standardized README.md files for every non-archived example from config.yaml metadata |
| `image.txt` | Base64-encoded test image used by `test_example.py` for image model inference |

### templating/

| Script | Description |
|--------|-------------|
| `generate_templates.py` | Programmatically generates truss configs for BEI, TEI, Briton, and BIS V2 deployments from model definitions |
| `deploy_all.py` | Bulk deploy/delete/test operations for generated templates (supports `--action deploy\|delete\|britontest` with `--filter`) |

### templates/

| File | Description |
|------|-------------|
| `generate.py` | Generates truss directories from base templates + config overrides defined in `generate.yaml` |
| `generate.yaml` | Declares legacy TRT-LLM and Transformers model variants (Mistral, Mixtral, Llama 2, Zephyr, Whisper) |
| `faster-whisper-truss/` | Base template for Faster Whisper models |
| `transformers-openai-compatible/` | Base template for HuggingFace Transformers with OpenAI-compatible API |
| `trt-llm/` | Base template for TRT-LLM models (includes Triton server configs) |

## Running the test suite

All scripts assume you are in the repository root.

```sh
# Run all local validation checks
python _internal/bin/test_all.py

# Verbose output (prints passing tests too)
python _internal/bin/test_all.py --verbose

# Filter to a single category
python _internal/bin/test_all.py --category llm
```

The test suite runs 11 checks:

1. Config validation (YAML parse + `truss.load()`)
2. README existence for every example
3. README-to-config consistency (endpoints, secrets, tags)
4. Directory naming conventions (hyphens, no underscores)
5. README link/path validation
6. `example_model_input` format validation
7. Requirements version pinning
8. CI path validation against `ci.yaml`
9. TRT-LLM OpenAI-compatible tag presence
10. `model.py` syntax validation (AST parse)
11. CI completeness (every example appears in `ci.yaml`)

### Other validation commands

```sh
# Structured validation report (writes validation_report.json)
python _internal/bin/validate_all.py
python _internal/bin/validate_all.py --json
python _internal/bin/validate_all.py --csv

# Quick CI path check
python _internal/bin/validate_ci.py
```

### README generation

```sh
# Regenerate all example READMEs from config.yaml metadata
python _internal/bin/generate_readmes.py
```

### Template generation (legacy)

```sh
cd _internal/templates
python generate.py --root ../.. --templates . --config generate.yaml

# Check-only mode (fails if generated output would differ)
python generate.py --only_check --root ../.. --templates . --config generate.yaml
```

## Dependencies

- **truss** -- required by all scripts. Install with `uv pip install -e ../truss` or `pip install truss`.
- **PyYAML** -- used for config parsing (typically installed with truss).
- `templating/generate_templates.py` additionally requires `transformers`, `pydantic`, and `requests`.
- `templating/deploy_all.py` requires `BASETEN_API_KEY` environment variable and optionally `openai`.
- `templates/generate.py` requires `jinja2` and `pydantic`.
