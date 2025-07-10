# Ultravox v0.6 70b vLLM Truss

Ultravox is a multimodal model that can consume both speech and text as input, generating output text as usual.

This is a [Truss](https://truss.baseten.co/) for Ultravox using the vLLM OpenAI Compatible server. This Truss bypasses the need for writing a `model.py` and instead runs `vllm serve` directly at startup and uses the HTTP endpoint provided by `vLLM` OpenAI Compatible Server to directly serve requests.

## OpenAI Bridge Compatibility

This Truss is compatible with a *custom* version of our [bridge endpoint for OpenAI ChatCompletion users](https://docs.baseten.co/api-reference/openai). This means you can easily integrate this model into your existing applications that use the OpenAI API format.

```
client = OpenAI(
    api_key=os.environ["BASETEN_API_KEY"],
    base_url=f"https://bridge.baseten.co/{model_id}/direct/v1"
)
```

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
truss push --publish --trusted
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
    base_url="https://bridge.baseten.co/v1/direct"
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
                    "audio_url": {"url": "https://audio-samples.github.io/samples/mp3/blizzard_tts_unbiased/sample-2/real.mp3"}
                }
            ]
        }
    ],
  extra_body={
    "baseten": {
      "model_id": model_id
    }
  }
)

print(response.choices[0].message.content)
```

## Support

If you have any questions or need assistance, please open an issue in this repository or contact our support team.
