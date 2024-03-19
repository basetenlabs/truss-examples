# Solar 10.7B Instruct V1.0 Truss

This is a [Truss](https://truss.baseten.co/) for [Solar 10.7B Instruct](https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0). This model often outperforms larger models which have more parameters. Solar 10.7B is also an ideal choice for fine-tuning as well.

## Truss
Truss is an open-source model serving framework developed by Baseten. It allows you to develop and deploy machine learning models onto Baseten (and other platforms like [AWS](https://truss.baseten.co/deploy/aws) or [GCP](https://truss.baseten.co/deploy/gcp). Using Truss, you can develop a GPU model using [live-reload](https://baseten.co/blog/technical-deep-dive-truss-live-reload), package models and their associated code, create Docker containers and deploy on Baseten.


## Deployment

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd solar-10.7b-instruct-v1.0
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `solar-10.7b-instruct-v1.0` as your working directory, you can deploy the model with:

```sh
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).


## Solar 10.7B Instruct API documentation

This section provides an overview of the Solar 10.7B Instruct model, its parameters, and how to use it. The API consists of a single route named  `predict`, which you can invoke to generate text based on the provided prompt.

### API route: `predict`

The predict route is the primary method for generating text completions based on a given prompt. It takes several parameters:

- __messages__: A list of JSON objects representing a conversation.
- __max_tokens__ (optional, default=4096): The maximum number of tokens to return, counting input tokens.
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
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {
        "role": "user",
        "content": "How much gold is too much gold?"
    }
]
```

The output of the model is a JSON object which only have one key called `output`. Here is an example of what that JSON object looks like:
```json
{"output": "### System:\nYou are a friendly chatbot who always responds in the style of a pirate\n\n### User:\nHow much gold is too much gold?\n\n### Assistant:\nAh, me hearty friend, that be a matter of perspective, as they say. Some say a thousand pieces be plenty to ensure a comfortable life for an honest sailor, while others crave an entire treasure grotto of sparklin' gold. It all depends on yer needs, desires, and the amount ye can safely stow away without attractin..."}
```

## Example usage

By default streaming the tokens is enabled. Here is an example of how to invoke the model with streaming in Python:
```python
import requests

headers = {"Authorization": f"Api-Key BASETEN-API-KEY"}
messages = [
    {"role": "system", "content": "You are a friendly chatbot who always responds in the style of a pirate"},
    {"role": "user", "content": "How much gold is too much gold?"}
]
data = {
    "messages": messages,
    "max_tokens": 512,
    "stream": True
}

res = requests.post(
    "https://model-<model-id>.api.baseten.co/development/predict",
    headers=headers,
    json=data,
    stream=True
)
res.raise_for_status()

for word in res:
    print(word.decode("utf-8"))
```
