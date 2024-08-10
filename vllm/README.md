# Truss for vLLM

This is a generic [Truss](https://truss.baseten.co/) that can

## OpenAI Bridge Compatibility

This Truss is compatible with a *custom* version of our [bridge endpoint for OpenAI ChatCompletion users](https://docs.baseten.co/api-reference/openai). This means you can easily integrate this model into your existing applications that use the OpenAI API format.

```
client = OpenAI(
    api_key=os.environ["BASETEN_API_KEY"],
    base_url=f"https://bridge.baseten.co/{model_id}/direct/v1"
)
```

## Truss

Truss is an open-source model serving framework developed by Baseten. It allows you to develop and deploy machine learning models onto Baseten (and other platforms like [AWS](https://truss.baseten.co/deploy/aws) or [GCP](https://truss.baseten.co/deploy/gcp)). Using Truss, you can develop a GPU model using [live-reload](https://baseten.co/blog/technical-deep-dive-truss-live-reload), package models and their associated code, create Docker containers, and deploy on Baseten.

## Deployment

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples.git
cd ultravox
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `ultravox` as your working directory, you can deploy the model with:

```sh
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

## vLLM OpenAI Compatible Server

This Truss demonstrates how to start [vLLM's OpenAI compatible server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html). The Truss is primarily used to start the server and then route requests to it. It currently supports ChatCompletions only.

### Passing startup arguments to the server

In the config any key-values under `model_metadata: arguments:` will be passed to the vLLM OpenAI-compatible server at startup.

### Base Image

You can use any vLLM compatible base image.

## API Documentation

The API follows the OpenAI ChatCompletion format. You can interact with the model using the standard ChatCompletion interface.

Example usage:

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR-API-KEY",
    base_url="https://bridge.baseten.co/MODEL-ID/v1"
)

response = client.chat.completions.create(
    model="fixie-ai/ultravox-v0.2",
    messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Summarize the following: <|audio|>"},
                {"type": "image_url", "image_url": {"url": f"data:audio/wav;base64,{base64_wav}"}}
            ]
        }]
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta)
```

## Future Improvements

We are actively working on enhancing this Truss. Some planned improvements include:

- Adding support for distributed serving with Ray (https://docs.vllm.ai/en/latest/serving/distributed_serving.html)
- Implementing model caching for improved performance

Stay tuned for updates!

## Support

If you have any questions or need assistance, please open an issue in this repository or contact our support team.
