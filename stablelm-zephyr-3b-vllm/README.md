# StableLM Zephyr 3B

This is a [Truss](https://truss.baseten.co/) for [StableLM Zephyr 3B](https://huggingface.co/stabilityai/stablelm-zephyr-3b), released by [Stability AI](https://stability.ai/). This README will walk you through how to deploy this Truss on Baseten to get your own instance of it.


## Deployment

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd stablelm-zerphyr-3b-vllm
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

You need a single A10G to run this model.

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
