# LLaMA-7B Truss

This is a [Truss](https://truss.baseten.co/) for an int8 version of LLaMA-7B. Llama is a family of language models released by Meta. This README will walk you through how to deploy this Truss on Baseten to get your own instance of LLaMA-7B.

## Truss

Truss is an open-source model serving framework developed by Baseten. It allows you to develop and deploy machine learning models onto Baseten (and other platforms like [AWS](https://truss.baseten.co/deploy/aws) or [GCP](https://truss.baseten.co/deploy/gcp). Using Truss, you can develop a GPU model using [live-reload](https://baseten.co/blog/technical-deep-dive-truss-live-reload), package models and their associated code, create Docker containers and deploy on Baseten.

## Deploying LLaMA-7B

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd llama-7b-truss
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `llama-7b-truss` as your working directory, you can deploy the model with:

```sh
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

## LLaMA-7B API documentation

This section provides an overview of the LLaMA-7B API, its parameters, and how to use it. The API consists of a single route named `predict`, which you can invoke to generate text based on the provided instruction.

### API route: `predict`

The predict route is the primary method for generating text completions based on a given instruction. It takes several parameters:

- **instruction**: The input text that you want the model to generate a response for.
- **temperature** (optional, default=0.1): Controls the randomness of the generated text. Higher values produce more diverse results, while lower values produce more deterministic results.
- **top_p** (optional, default=0.75): The cumulative probability threshold for token sampling. The model will only consider tokens whose cumulative probability is below this threshold.
- **top_k** (optional, default=40): The number of top tokens to consider when sampling. The model will only consider the top_k highest-probability tokens.
- **num_beams** (optional, default=4): The number of beams used for beam search. Increasing this value can result in higher-quality output but will increase the computational cost.

The API also supports passing any parameter supported by Huggingface's `Transformers.generate`.

## Example usage

```sh
truss predict -d '{"prompt": "What is the meaning of life?"}'
```

You can also invoke your model via a REST API

```sh
curl -X POST " https://app.baseten.co/models/YOUR_MODEL_ID/predict" \
     -H "Content-Type: application/json" \
     -H 'Authorization: Api-Key {YOUR_API_KEY}' \
     -d '{
           "prompt": "What's the meaning of life?",
           "temperature": 0.1,
           "top_p": 0.75,
           "top_k": 40,
           "num_beams": 4
         }'

```
