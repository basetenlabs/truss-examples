# MPT-7B Base Truss

This is a [Truss](https://truss.baseten.co/) for [MPT-7B](https://www.mosaicml.com/blog/mpt-7b) using the `mosaicml/mpt-7b` family of models from the `transformers` library. This README will walk you through how to deploy the base version on Baseten.

## Truss

Truss is an open-source model serving framework developed by Baseten. It allows you to develop and deploy machine learning models onto Baseten (and other platforms like [AWS](https://truss.baseten.co/deploy/aws) or [GCP](https://truss.baseten.co/deploy/gcp). Using Truss, you can develop a GPU model using [live-reload](https://baseten.co/blog/technical-deep-dive-truss-live-reload), package models and their associated code, create Docker containers and deploy on Baseten.

## Deploying MPT-7B

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd mpt-7b-base-truss
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `mpt-7b-base-truss` as your working directory, you can deploy the model with:

```sh
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

## MPT-7B API documentation
This section provides an overview of the MPT-7B Base API, its parameters, and how to use it. The API consists of a single route named `predict`, which you can invoke to generate text completions based on the provided parameters.

### API route: `predict`
The predict route is the primary method for generating images based on a given set of parameters. It takes several parameters:

- __prompt__: The input text prompt for the LLM
- __max_tokens__: (optional) The maximum number of tokens that will be generated
- __temperature__: (optional) A temperature of 0 means the response is more deterministic. A temperature of greater than zero results in increasing variation in the completion.
- __top_k__: (optional) Controls how model picks the next token from the top `k` tokens in its list, sorted by probability.
- __top_p__: (optional) Controls how the model picks from the top tokens based on the sum of their probabilities.

## Example usage

You can use the `baseten` model package to invoke your model from Python

```sh
truss predict -d '{"prompt" : "Today I inspected the engine mounting equipment. I found a problem in one of the brackets so"}'
```

You can also invoke your model via a REST API
```
curl -X POST " https://app.baseten.co/models/YOUR_MODEL_ID/predict" \
     -H "Content-Type: application/json" \
     -H 'Authorization: Api-Key {YOUR_API_KEY}' \
     -d '{
    "prompt" : "Today I inspected the engine mounting equipment. I found a problem in one of the brackets so" \
    "temperature": 0.75, \
    "max_tokens": 200 \
}'
```
