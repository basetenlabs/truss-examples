# Ultravox v0.6 70B vLLM Truss

Ultravox is a multimodal model that can consume both speech and text as input, generating output text as usual.

This is a [Truss](https://truss.baseten.co/) for Ultravox using the vLLM OpenAI Compatible server. This Truss bypasses the need for writing a `model.py` and instead runs `vllm serve` directly at startup and uses the HTTP endpoint provided by `vLLM` OpenAI Compatible Server to directly serve requests.

## Deployment

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples.git
cd custom-server/ultravox-0.6-70b
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`
3. Retrieve your Hugging Face token from the [settings](https://huggingface.co/settings/tokens).
4. Set your Hugging Face token as a Baseten secret [here](https://app.baseten.co/settings/secrets) with the key `hf_access_key`. Note that you will *not* be able to successfully deploy Ultravox without doing this.

With `ultravox-0.6-70b` as your working directory, you can deploy the model with:

```sh
truss push --publish
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

## vLLM OpenAI Compatible Server

This Truss demonstrates how to start [vLLM's OpenAI compatible server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) without the need for a `model.py` through the `docker_server.start_command` option.

## API Documentation

The API follows the OpenAI ChatCompletion format. You can interact with the model using the standard ChatCompletion interface.

Example usage:

```python
from openai import OpenAI

model_id = "jwdp26kw" # Replace with your model ID

client = OpenAI(
    api_key="YOUR-API-KEY",
    base_url=f"https://model-{model_id}.api.baseten.co/environments/production/sync/v1"
)

response = client.chat.completions.create(
    model="ultravox", # Replace with your model name
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is Lydia like?"
                },
                {
                    "type": "audio_url",
                    "audio_url": {"url": "https://baseten-public.s3.us-west-2.amazonaws.com/fred-audio-tests/real.mp3"}
                }
            ]
        }
    ],
    stream=True
)

for chunk in response:
    content = chunk.choices[0].delta.content
    print(content, end="", flush=True)
```

## Support

If you have any questions or need assistance, please open an issue in this repository or contact our support team.
