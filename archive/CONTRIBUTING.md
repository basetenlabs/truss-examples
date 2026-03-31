# Contributing

Please open a PR to add a new model. Please add new models to the `model_library` folder.

## Style guide

Model library models should follow this style guide.

General style tips:

* Truss folder name should use hyphens, not underscores (e.g. `falcon-7b` not `falcon_7b`)
* Truss folder name should not include the word Truss (e.g. `falcon-7b` not `falcon-7b-truss`)
* Model name should include parameter count when multiple variants exist (e.g. `Falcon 7B` not `Falcon`)

## README

The model should have a README that follows the layout of [Stable Diffusion XL](stable-diffusion/stable-diffusion-xl-1.0).


## Config

Do not include any unnecessary or legacy config lines.

Always include:

* `model_name`
* `description`
* `model_metadata`
  * `example_model_input`

Optionally include:
* `model_metadata`
  * `avatar_url` (Avatars/Logos should be 128x128 PNG)
  * `cover_image_url` (Cover images should be 452x423 PNG)
  * `tags`

### Requirements

Pin versions for all Python requirements!

Example:

```yaml
requirements:
- accelerate==0.20.3
- bitsandbytes==0.39.1
- peft==0.3.0
- protobuf==4.23.3
- sentencepiece==0.1.99
- torch==2.0.1
- transformers==4.30.2
```

### Secrets

Model library models can access secrets.

If the model requires HuggingFace (e.g. Llama 2), always call the secret `hf_access_token`

Example:

```yaml
secrets:
  hf_access_token: "ENTER HF ACCESS TOKEN HERE"
```

### Hardware requirements

Always configure a model library model with the least expensive hardware required to operate it at a reasonable degree of speed and quality. For example, Stable Diffusion XL defaults to an A10 even though performance is twice as fast on an A100. When these tradeoffs are made, note them in the README.

## Model

### Model I/O

* Models that support streaming should take a `stream` kwarg that defaults to false
* Models that take any kind of text input should call it `prompt`

# Testing

If you would like the model to be added the CI job that tests examples very day, add a reference
to the [ci.yaml](ci.yaml) file.

# Automatic Documentation

Some of the examples in this repo are used to generate automatic documentation on https://truss.baseten.co/.

To add your model to this automatic documentation, add your example to one of the **top-level** categories
in the repo, and add a `doc.yaml` file that follows the following form:

```yaml
title: "Text-to-image"
description: "Building a text-to-image model with SDXL"
files:
  - model/model.py
  - config.yaml
```

See the [Introduction doc.yaml file](1_introduction/getting-started-bert/doc.yaml) for an example.
