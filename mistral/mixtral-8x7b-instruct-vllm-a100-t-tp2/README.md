# Mixtral 8x7B Instruct Truss

This is a [Truss](https://truss.baseten.co/) for Mixtral 8x7B Instruct. Mixtral 8x7B Instruct parameter language model released by [Mistral AI](https://mistral.ai/). It is a mixture-of-experts (MoE) model. This README will walk you through how to deploy this Truss on Baseten to get your own instance of it.


## Deployment

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd mixtral-8x7b-instruct-vllm
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `mixtral-8x7b-instruct-vllm` as your working directory, you can deploy the model with:

```sh
truss push --publish
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

### Hardware notes

You need two A100s to run Mixtral at `fp16`. If you need access to A100s, please [contact us](mailto:support@baseten.co).

## Mixtral 8x7B Instruct API documentation

This section provides an overview of the Mixtral 8x7B Instruct API, its parameters, and how to use it. The API consists of a single route named  `predict`, which you can invoke to generate text based on the provided prompt.

### API route: `predict`

The `predict` route is the primary method for generating text completions based on a given prompt. It takes several parameters:

- __prompt__: The input text that you want the model to generate a response for.
- __stream__ (optional, default=False): A boolean determining whether the model should stream a response back. When `True`, the API returns generated text as it becomes available.

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
           "prompt": "What is the meaning of life? Answer in substantial detail with multiple examples from famous philosophies, religions, and schools of thought.",
           "stream": true,
           "max_tokens": 4096
         }' --no-buffer
```
