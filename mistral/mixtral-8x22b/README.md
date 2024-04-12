# Mixtral 8x22B Truss

This is a [Truss](https://truss.baseten.co/) for the community edition of [Mixtral 8x22B](https://huggingface.co/mistral-community/Mixtral-8x22B-v0.1).
This is not an optimized model. If you would like to have a more optimized version that has lower latency + higher throughput, please contact our team. 


## Deployment

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd mistral/mixtral-8x22b
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `mixtral-8x22b` as your working directory, you can deploy the model with:

```sh
truss push --publish
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

### Hardware notes

You need four A100s to run Mixtral at `fp16`. If you need access to A100s, please [contact us](mailto:support@baseten.co).

## Mixtral 8x22B API documentation

This section provides an overview of the Mixtral 8x22B API, its parameters, and how to use it. The API consists of a single route named  `predict`, which you can invoke to generate text based on the provided prompt.

### API route: `predict`

The `predict` route is the primary method for generating text completions based on a given prompt. It takes several parameters:

- __prompt__: The input text that you want the model to generate a response for.
- __stream__ (optional, default=True): A boolean determining whether the model should stream a response back. When `True`, the API returns generated text as it becomes available.
- __max_tokens__ (optional, default=128): Determines the maximum number of tokens to generate
- __temperature__ (optional, default=1.0): Controls the strength of the generation. The higher the temperature, the more diverse and creative the output would be.
- __top_p__ (optional, default=0.95): Parameter used to control the randomness of the output.
- __top_k__ (optional, default=50): Controls the vocab size considered during the generation.

## Example usage

```python
import requests
import os

# Replace the empty string with your model id below
model_id = ""
baseten_api_key = os.environ["BASETEN_API_KEY"]

data = {
    "prompt": "What is mistral wind?",
    "stream": True,
    "max_tokens": 256,
    "temperature": 0.9
}

# Call model endpoint
res = requests.post(
    f"https://model-{model_id}.api.baseten.co/production/predict",
    headers={"Authorization": f"Api-Key {baseten_api_key}"},
    json=data,
    stream=True
)

# Print the generated tokens as they get streamed
for content in res.iter_content():
    print(content.decode("utf-8"), end="", flush=True)
```
