[![Deploy to Baseten](https://user-images.githubusercontent.com/2389286/236301770-16f46d4f-4e23-4db5-9462-f578ec31e751.svg)](https://app.baseten.co/explore/alpaca)

# Alpaca-7B Truss

This is a [Truss](https://truss.baseten.co/) for Alpaca-7B, a fine-tuned variant of LLaMA-7B. LLaMA is a family of language models released by Meta. This README will walk you through how to deploy this Truss on Baseten to get your own instance of Alpaca-7B.
## Deploy Alpaca-7B

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd alpaca-7b-truss
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `alpaca-7b-truss` as your working directory, you can deploy the model with:

```sh
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

## Alpaca-7B API documentation
This section provides an overview of the Alpaca-7B API, its parameters, and how to use it. The API consists of a single route named  `predict`, which you can invoke to generate text based on the provided instruction.

### API route: `predict`
The predict route is the primary method for generating text completions based on a given instruction. It takes several parameters:

- __instruction__: The input text that you want the model to generate a response for.
- __temperature__ (optional, default=0.1): Controls the randomness of the generated text. Higher values produce more diverse results, while lower values produce more deterministic results.
- __top_p__ (optional, default=0.75): The cumulative probability threshold for token sampling. The model will only consider tokens whose cumulative probability is below this threshold.
- __top_k__ (optional, default=40): The number of top tokens to consider when sampling. The model will only consider the top_k highest-probability tokens.
- __num_beams__ (optional, default=4): The number of beams used for beam search. Increasing this value can result in higher-quality output but will increase the computational cost.

The API also supports passing any parameter supported by Huggingface's `Transformers.generate`.

## Example usage

```sh
truss predict -d '{"prompt": "What's the meaning of life?"}'
```

You can also invoke your model via a REST API:

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
