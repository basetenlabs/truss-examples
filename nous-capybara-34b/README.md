# Nous Capybara 34B Truss

This is a [Truss](https://truss.baseten.co/) for [Nous Capybara 34B](https://huggingface.co/NousResearch/Nous-Capybara-34B). This model is a fine-tuned version of Yi-34B with a 200K context length.

## Truss
Truss is an open-source model serving framework developed by Baseten. It allows you to develop and deploy machine learning models onto Baseten (and other platforms like [AWS](https://truss.baseten.co/deploy/aws) or [GCP](https://truss.baseten.co/deploy/gcp). Using Truss, you can develop a GPU model using [live-reload](https://baseten.co/blog/technical-deep-dive-truss-live-reload), package models and their associated code, create Docker containers and deploy on Baseten.


## Deployment

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd nous-capybara-34b
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `nous-capybara-34b` as your working directory, you can deploy the model with:

```sh
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).


### Hardware notes

This 34 billion parameter model requires an A100 GPU.

## Nous Capybara 34B Chat API documentation

This section provides an overview of the Nous-Capybara-34B model, its parameters, and how to use it. The API consists of a single route named  `predict`, which you can invoke to generate text based on the provided prompt.

### API route: `predict`

The predict route is the primary method for generating text completions based on a given prompt. It takes several parameters:

- __prompt__: The input text that you want the model to generate a response for.
- __max_tokens__ (optional, default=256): The maximum number of tokens to return, counting input tokens.
- __temperature__ (optional, default=0.7): Controls the randomness of the generated text. Higher values produce more diverse results, while lower values produce more deterministic results.
- __top_p__ (optional, default=0.8): The cumulative probability threshold for token sampling. The model will only consider tokens whose cumulative probability is below this threshold.
- __top_k__ (optional, default=40): The number of top tokens to consider when sampling. The model will only consider the top_k highest-probability tokens.
- __repetition_penalty__ (optional, default=1.3): Helps the model generate more diverse content instead of repeating previous phrases.
- __no_repeat_ngram_size__ (optional, default=5): Specifies the length of token sets that are completely blocked from repeating at all.
- __stream__ (optional, default=True): Allows you to receive the tokens are they are generated in a stream like fashion.

The API also supports passing any parameter supported by HuggingFace's `Transformers.generate`.

The output of the model is a JSON object which only have one key called `output`. Here is an example of what that JSON object looks like:
```json
{"output": "There's a place where time stands still. A place of breath taking wonder, but also deep mystery and danger; the ocean floor.."}
```

## Example usage

```sh
truss predict -d '{"prompt": "There is a place where time stands still. A place of breath taking wonder, but also", "max_tokens": 512}'
```

You can also invoke your model via a REST API:

```
curl -X POST " https://app.baseten.co/model_versions/YOUR_MODEL_VERSION_ID/predict" \
     -H "Content-Type: application/json" \
     -H 'Authorization: Api-Key {YOUR_API_KEY}' \
     -d '{
           "prompt": "What happens if I go to the top of the tallest mountian in california with a bucket of water and tip it over the highest cliff?",
           "max_tokens": 512,
           "stream": False
         }'
```

By default streaming the tokens is enabled. Here is an example of how to invoke the model with streaming in Python:
```python
import requests
headers = {"Authorization": f"Api-Key BASETEN-API-KEY"}

res = requests.post(
    "https://model-<model-id>.api.baseten.co/development/predict",
    headers=headers,
    json={"prompt": "What happens if I go to the top of the tallest mountian in california with a bucket of water and tip it over the highest cliff?",
          "max_tokens": 512, "temperature": 0.9, "stream": True},
    stream=True
)
res.raise_for_status()

for word in res:
    print(word.decode("utf-8"))
```
