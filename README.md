# Truss Examples

[![Truss Examples CI](https://github.com/basetenlabs/truss-examples/actions/workflows/test-examples.yml/badge.svg)](https://github.com/basetenlabs/truss-examples/actions/workflows/test-examples.yml)

Truss is the simplest way to serve AI/ML models in production.

To get you started with [Truss](https://truss.baseten.co/), this repository has dozens of example models, each ready to deploy as-is or adapt to your needs.

## Installation

Get the repository with:

```
git clone https://github.com/basetenlabs/truss-examples
```

Install Truss with:

```
pip install --upgrade truss
```

## Deployment


Pick a model to deploy by passing a path to that model.

```bash
$ # From the truss-examples directory
$ truss push 02-llm
```

This will prompt you for an API Key -- fetch one from the
[Baseten API keys page](https://app.baseten.co/settings/account/api_keys).

## Invocation

Invocation depends on the model's input and output specifications. See individual model READMEs for invocation details.

# Contibuting

We welcome contributions of new models and improvements to existing models. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.
