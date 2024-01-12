 # All MPNet Base V2 Truss

This repository packages [All MPNet Base V2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) as a [Truss](https://truss.baseten.co).


## Deploying All MPNet Base V2

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd all-mpnet-base-v2
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `all-mpnet-base-v2` as your working directory, you can deploy the model with:

```sh
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

## Invoking All MPNet Base V2

The model takes a dictionary with:

* `text`: A list of strings. Each string will be encoded into a text embedding and returned.

Example invocation:

```sh
truss predict -d '{"text": ["I want to eat pasta", "I want to eat pizza"]}'
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

## Hardware notes

For creating a few embeddings from relatively short chunks of text, a CPU-only instance with 4 cores and 16 GiB of RAM is more than sufficient. If you need to quickly create an embedding for a large corpus of text, you may want to upgrade to a larger instance type or add a small GPU like an NVIDIA T4.

Default config:

```yaml
...
resources:
  cpu: "4"
  memory: 16Gi
  use_gpu: false
  accelerator: null
...
```
