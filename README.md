# Truss Examples

Truss is the simplest way to serve AI/ML models in production. 

To get you started with [Truss](https://truss.baseten.co/), this repository has dozens of example models, each ready to deploy as-is or adapt to your needs.

# Using this repository

The top-level directories (such as `2_image_classification`) contain high-level categories, and contain
well-commented examples of models in that category.  Each of these is tested with a CI job once a day.

There are also lots of other model examples in the `model_library`, what likely has something similar
to what you're looking for! 

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
$ truss push 2_image_classification/clip
```

This will prompt you for an API Key -- fetch one from the
[Baseten API keys page](https://app.baseten.co/settings/account/api_keys).

## Invocation

Invocation depends on the model's input and output specifications. See individual model READMEs for invocation details.

# Contibuting

We welcome contributions of new models and improvements to existing models. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

