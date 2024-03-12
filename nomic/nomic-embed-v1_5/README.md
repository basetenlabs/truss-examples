 # Nomic Embed v1.5 Truss

This repository packages [Nomic Embed 1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) as a [Truss](https://truss.baseten.co).

This text embedding model has a flexible dimensionality and best in class performance.

## Deploying Nomic Embed v1.5

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd nomic/nomic-embed-v1_5
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `nomic/nomic-embed-v1_5` as your working directory, you can deploy the model with:

```sh
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

## Invoking Nomic Embed v1.5

The model takes a dictionary with:

* `text`: A list of strings. Each string will be encoded into a text embedding and returned.
* `dimensionality` (optional): The output dimension of the embedding, in range [64, 768], default 768. Higher dimensionality means better performance, lower dimensionality means faster inference and cheaper storage and retrieval.

Example invocation:

```sh
truss predict -d '{"text": ["I want to eat pasta", "I want to eat pizza"], "dimensions": 768}'
```

Expected response:

```python
[
  [
    0.2593194842338562,
    ...
    -1.4059709310531616
  ],
  [
    0.11028853803873062,
    ...
    -0.9492666125297546
  ],
]
```

We also prepared a sample input file of all 154 of Shakespeare's sonnets. You can create embeddings from this file with:

```sh
truss predict -f sample.json > output.json
```
