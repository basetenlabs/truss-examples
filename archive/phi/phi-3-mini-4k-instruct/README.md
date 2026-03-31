# Phi-3-Mini-4K-Instruct

This is a [Truss](https://truss.baseten.co/) for Phi-3-Mini-4K-Instruct. This README will walk you through how to deploy this Truss on Baseten to get your own instance of Llama 3 8B.

## Deployment

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd phi/phi-3-mini-4k-instruct
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `phi/phi-3-mini-4k-instruct` as your working directory, you can deploy the model with:

```sh
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

## Phi-3-Mini-4K-Instruct API documentation

This section provides an overview of the Phi-3-Mini-4K-Instruct API, its parameters, and how to use it. The API consists of a single route named  `predict`, which you can invoke to generate text based on the provided prompt.

### API route: `predict`

The predict route is the primary method for generating text completions based on a given prompt. It takes several parameters:

- __messages__: The input text that you want the model to generate a response for.
- __max_tokens__ (optional, default=512): The maximum number of tokens to return, counting input tokens. Maximum of 4096.
- __temperature__ (optional, default=1.0): Controls the randomness of the generated text. Higher values produce more diverse results, while lower values produce more deterministic results.
- __top_p__ (optional, default=0.75): The cumulative probability threshold for token sampling. The model will only consider tokens whose cumulative probability is below this threshold.
- __top_k__ (optional, default=40): The number of top tokens to consider when sampling. The model will only consider the top_k highest-probability tokens.

The API also supports passing any parameter supported by HuggingFace's `Transformers.generate`.

## Example usage

You can also invoke your model via a REST API:

```
curl -X POST " https://app.baseten.co/model_versions/YOUR_MODEL_VERSION_ID/predict" \
     -H "Content-Type: application/json" \
     -H 'Authorization: Api-Key {YOUR_API_KEY}' \
     -d '{
           "messages": [{"role": "user", "content": "What even is AGI?"}],
           "max_tokens": 256
         }'
```
