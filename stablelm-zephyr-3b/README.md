# StableLM Zephyr 3B

This is a [Truss](https://truss.baseten.co/) for StableLM Zephyr 3B.

## Truss
Truss is an open-source model serving framework developed by Baseten. It allows you to develop and deploy machine learning models onto Baseten (and other platforms like [AWS](https://truss.baseten.co/deploy/aws) or [GCP](https://truss.baseten.co/deploy/gcp). Using Truss, you can develop a GPU model using [live-reload](https://baseten.co/blog/technical-deep-dive-truss-live-reload), package models and their associated code, create Docker containers and deploy on Baseten.


## Deployment

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd stablelm-zephyr-3b
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `stablelm-zephyr-3b` as your working directory, you can deploy the model with:

```sh
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).


## StableLM Zephyr 3B API documentation

This section provides an overview of the StableLM Zephyr 3B model, its parameters, and how to use it. The API consists of a single route named  `predict`, which you can invoke to generate text based on the provided prompt.

### API route: `predict`

The predict route is the primary method for generating text completions based on a given prompt. It takes several parameters:

- __messages__: A list of JSON objects representing a conversation.
- __max_new_tokens__ (optional, default=4096): The maximum number of tokens to return, counting input tokens.
- __temperature__ (optional, default=1.0): Controls the randomness of the generated text. Higher values produce more diverse results, while lower values produce more deterministic results.
- __top_p__ (optional, default=0.95): The cumulative probability threshold for token sampling. The model will only consider tokens whose cumulative probability is below this threshold.
- __top_k__ (optional, default=40): The number of top tokens to consider when sampling. The model will only consider the top_k highest-probability tokens.
- __repetition_penalty__ (optional, default=1.0): Helps the model generate more diverse content instead of repeating previous phrases.
- __no_repeat_ngram_size__ (optional, default=0): Specifies the length of token sets that are completely blocked from repeating at all.
- __stream__ (optional, default=True): Allows you to receive the tokens are they are generated in a stream like fashion.

The API also supports passing any parameter supported by HuggingFace's `Transformers.generate`.

Here is an example of the `messages` the model takes as input:
```json
[
    {
        "role": "user",
        "content": "How much gold is too much gold?"
    },
    {
        "role": "assistant",
        "content": "There is no such thing as too much gold."
    }
]
```

The output of the model is a JSON object which only have one key called `output`. Here is an example of what that JSON object looks like:
```json
{"output": "<|user|>\nWhat is a zephyr?\n<|assistant|>\nA zephyr is a gentle, cool breeze that usually blows from the mountains or highland regions.\n<|user|>\nWhat is the speed in MPH for a zephyr?\n<|assistant|>\n..."}
```

## Example usage

By default streaming the tokens is enabled. Here is an example of how to invoke the model with streaming in Python:
```python
import requests

headers = {"Authorization": f"Api-Key BASETEN-API-KEY"}
messages = [
    {"role": "user", "content": "What is a zephyr?"},
    {"role": "assistant", "content": "A zephyr is a gentle, cool breeze that usually blows from the mountains or highland regions."},
    {"role": "user", "content": "What is the speed in MPH of a zephyr?"},
]
data = {
    "messages": messages,
    "max_new_tokens": 1024,
    "stream": True
}

res = requests.post(
    "https://model-<model-id>.api.baseten.co/development/predict",
    headers=headers,
    json=data,
    stream=True
)

for content in res.iter_content():
    print(content.decode("utf-8"), end="", flush=True)
```
