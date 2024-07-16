# Gemma 2 27B

This is a [Truss](https://truss.baseten.co/) for Gemma 2 27B Instruct. This README will walk you through how to deploy this Truss on Baseten to get your own instance of Gemma 2 27B Instruct.

## Gemma 2 27B Instruct Implementation

This implementation of Gemma 2 uses [vLLM](https://github.com/vllm-project/vllm).

Since Gemma 2 is a gated model, you will also need to provide your Huggingface access token after making sure you have access to [the model](https://huggingface.co/google/gemma-2-27b-it). Please use the [following guide](https://docs.baseten.co/deploy/guides/secrets) to add your Huggingface access token as a secret.

## Deployment

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd gemma2/gemma2-27b-it-vllm
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `gemma2/gemma2-27b-it-vllm` as your working directory, you can deploy the model with:

```sh
truss push --trusted
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

## Gemma 2 27B Instruct API documentation

This section provides an overview of the Gemma 2 27B Instruct API, its parameters, and how to use it. The API consists of a single route named  `predict`, which you can invoke to generate text based on the provided prompt.

### API route: `predict`

The predict route is the primary method for generating text completions based on a given prompt. It takes several parameters:

- __prompt__: The input text that you want the model to generate a response for.
- __max_tokens__: The maximum number of output tokens.

## Example usage

You can also invoke your model via a REST API:

```
curl -X POST " https://app.baseten.co/model_versions/YOUR_MODEL_VERSION_ID/predict" \
     -H "Content-Type: application/json" \
     -H 'Authorization: Api-Key {YOUR_API_KEY}' \
     -d '{"prompt": "what came before, the chicken or the egg?", "max_tokens": 64}'
```