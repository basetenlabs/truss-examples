# Contributing

We welcome new models and improvements to existing examples. Please open a PR.

## Adding a new example

1. Place your example in the category that matches its type (`llm/`, `image/`, `audio/`, `embeddings/`, `infrastructure/`, or `tutorials/`).
2. Include a `config.yaml` with `model_name`, `description`, and `example_model_input`.
3. Include a `README.md` with deploy and invoke instructions.
4. Pin all Python requirements to specific versions.
5. Add your example path to `ci.yaml` at the repo root.

## Validate locally

```bash
python _internal/bin/test_all.py
```

This runs all config, README, naming, and CI checks. Pre-commit hooks run the same suite automatically.

## Questions?

If your model doesn't fit an existing category, open an issue to discuss placement.
