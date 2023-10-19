# Mistral 7B Instruct Truss

This is a [Truss](https://truss.baseten.co/) for Mistral 7B Instruct. Mistral 7B Instruct parameter language model released by [Mistral](https://mistral.ai/) that outperforms other models in the 7B model class. This README will walk you through how to deploy this Truss on Baseten to get your own instance of Mistral 7B Instruct.

## Truss

Truss is an open-source model serving framework developed by Baseten. It allows you to develop and deploy machine learning models onto Baseten (and other platforms like [AWS](https://truss.baseten.co/deploy/aws) or [GCP](https://truss.baseten.co/deploy/gcp)). Using Truss, you can develop a GPU model using [live-reload](https://baseten.co/blog/technical-deep-dive-truss-live-reload), package models and their associated code, create Docker containers and deploy on Baseten.

## Deployment

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd mistral-7b-instruct
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `mistral-7b-instruct` as your working directory, you can deploy the model with:

```sh
truss push --publish
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

### Hardware notes

This seven billion parameter model is running in `float16` so that it fits on an A10G.

## Mistral 7B Instruct API documentation

This section provides an overview of the Mistral 7B Instruct API, its parameters, and how to use it. The API consists of a single route named  `predict`, which you can invoke to generate text based on the provided prompt.

### API route: `predict`

The `predict` route is the primary method for generating text completions based on a given prompt. It takes several parameters:

- __prompt__: The input text that you want the model to generate a response for.
- __n__ (optional, default=1): The number of sequences to return, selects from a subset of `best_of`
- __best_of__ (optional, default=1): The number of sequences to generate
- __max_tokens__ (optional, default=512): The maximum number of tokens to return, counting input tokens. Maximum of 4096.
- __temperature__ (optional, default=1.0): Controls the randomness of the generated text. Higher values produce more diverse results, while lower values produce more deterministic results.
- __top_p__ (optional, default=0.95): The cumulative probability threshold for token sampling. The model will only consider tokens whose cumulative probability is below this threshold.
- __top_k__ (optional, default=50): The number of top tokens to consider when sampling. The model will only consider the top_k highest-probability tokens.
- __frequency_penalty__ (optional, default=1.0): Controls the model’s penalty on producing the same token sequence, with higher values discouraging repetition.
- __prescence_penalty__ (optional, default=1.0): Controls the model’s penalty on producing the same token if it's already been produced. 
- __use_beam_search__ (optional, default=True): Controls the sampling strategy during the decoding process. Setting to False results in the generation process sampling the highest probability next token (greedy decoding). Otherwise, we sample non-greedily via multinomial sampling. Define `n` and `best_of` if this is set to True.

## Example usage

```sh
truss predict -d '{"prompt": "What is the Mistral wind?"}'
```

You can also invoke your model via a REST API:

```
curl -X POST " https://app.baseten.co/model_versions/YOUR_MODEL_VERSION_ID/predict" \
     -H "Content-Type: application/json" \
     -H 'Authorization: Api-Key {YOUR_API_KEY}' \
     -d '{
           "What is the meaning of life? Answer in substantial detail with multiple examples from famous philosophies, religions, and schools of thought.",
           "max_tokens": 4096
         }' --no-buffer
```
