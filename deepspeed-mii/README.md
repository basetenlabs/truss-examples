# Llama-2-chat 7B DeepSpeed MII Truss

This is a [Truss](https://truss.baseten.co/) for Llama-2-chat 7B served with [DeepSpeed MII](https://github.com/microsoft/DeepSpeed-MII). Llama 2 is a family of language models released by Meta. This README will walk you through how to deploy this Truss on Baseten to get your own instance of Llama-2-chat 7B.

## Truss

Truss is an open-source model serving framework developed by Baseten. It allows you to develop and deploy machine learning models onto Baseten (and other platforms like [AWS](https://truss.baseten.co/deploy/aws) or [GCP](https://truss.baseten.co/deploy/gcp)). Using Truss, you can develop a GPU model using [live-reload](https://baseten.co/blog/technical-deep-dive-truss-live-reload), package models and their associated code, create Docker containers and deploy on Baseten.

### Get Llama 2 access

Llama 2 currently requires approval to access. To request access:

1. Go to [https://ai.meta.com/resources/models-and-libraries/llama-downloads/](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and request access using the email associated with your HuggingFace account.
2. Go to [https://huggingface.co/meta-llama/Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b) and request access.

Once you have Llama access:

1. Create a [HuggingFace access token](https://huggingface.co/settings/tokens)
2. Set it as a [secret in your Baseten account](https://app.baseten.co/settings/secrets) with the name `hf_access_token`

## Deployment

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd deepspeed-mii
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `deepspeed-mii` as your working directory, you can deploy the model with:

```sh
truss push --trusted
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

### Hardware notes

This seven billion parameter model is running in `float16` so that it fits on an A10G.

## Llama-2-chat 7B API documentation

This section provides an overview of the Llama-2-chat 7B API, its parameters, and how to use it. The API consists of a single route named  `predict`, which you can invoke to generate text based on the provided prompt.

### API route: `predict`

The predict route is the primary method for generating text completions based on a given prompt. It takes several parameters:

- __prompt__: The input text that you want the model to generate a response for.
- __max_length__ (optional, default=512): The maximum number of tokens to return, counting input tokens. Maximum of 4096.

## Example usage

```sh
truss predict -d '{"prompt": "What is the meaning of life?", "max_length": 1024}'
```

You can also invoke your model via a REST API:

```
curl -X POST " https://app.baseten.co/model_versions/YOUR_MODEL_VERSION_ID/predict" \
     -H "Content-Type: application/json" \
     -H 'Authorization: Api-Key {YOUR_API_KEY}' \
     -d '{
           "prompt": "What's the meaning of life?",
           "max_length": 1024
         }'
```
