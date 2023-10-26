 # Jina Embeddings V2 Base EN Truss

This repository packages [Jina Embeddings V2 Base EN](https://huggingface.co/jinaai/jina-embeddings-v2-base-en) as a [Truss](https://truss.baseten.co).

This text embedding model has a context window of 8,192 tokens and performs comparably to OpenAI's ada-002 text embedding model on standard benchmarks.

## Deploying Jina Embeddings V2 Base EN

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd jina-embeddings/jina-embeddings-v2-base-en
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `jina-embeddings-v2-base-en` as your working directory, you can deploy the model with:

```sh
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

## Invoking Jina Embeddings V2 Base EN

The model takes a dictionary with:

* `text`: A list of string. Each string will be encoded into a text embedding and returned.
* `max_length` (optional): The number of tokens per string to encode. Default is 8,192, which is also the maximum number of tokens the model can process per string.

Example invocation:

```sh
truss predict -d '{"text": ["I want to eat pasta", "I want to eat pizza"], "max_length": 8192}'
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

For creating a few embeddings from relatively short chunks of text, a CPU-only instance with 4 cores and 16 GiB of RAM is more than sufficient.

On that instance type, creating an embedding for each of Shakespeare's 154 sonnets (just under 100KB of text) takes about a minute and a half. If you need to quickly create an embedding for a large corpus of text, you may want to upgrade to a larger instance type or add a small GPU like an NVIDIA T4.

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
