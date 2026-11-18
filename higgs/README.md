# Higgs Audio v2 Generation 3B vLLM Truss

Higgs Audio v2 Generation 3B is a multimodal model that can generate audio content based on text and audio inputs.

This is a [Truss](https://truss.baseten.co/) for Higgs Audio using the vLLM OpenAI Compatible server. This Truss bypasses the need for writing a `model.py` and instead runs `vllm serve` directly at startup and uses the HTTP endpoint provided by `vLLM` OpenAI Compatible Server to directly serve requests.

## Deployment

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples.git
cd higgs
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`
3. Retrieve your Hugging Face token from the [settings](https://huggingface.co/settings/tokens).
4. Set your Hugging Face token as a Baseten secret [here](https://app.baseten.co/settings/secrets) with the key `hf_access_token`. Note that you will *not* be able to successfully deploy Higgs Audio without doing this.

With `higgs` as your working directory, you can deploy the model with:

```sh
truss push --publish
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

## vLLM OpenAI Compatible Server

This Truss demonstrates how to start [vLLM's OpenAI compatible server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) without the need for a `model.py` through the `docker_server.start_command` option.

The server is configured with the following parameters:
- **Model**: `bosonai/higgs-audio-v2-generation-3B-base`
- **Audio Tokenizer**: `bosonai/higgs-audio-v2-tokenizer`
- **Max Model Length**: 8192 tokens
- **GPU Memory Utilization**: 80%
- **Audio Limit**: 50 audio inputs per prompt
- **MM Preprocessor Cache**: Disabled for better performance

## API Documentation

The API follows the OpenAI ChatCompletion format. You can interact with the model using the standard ChatCompletion interface.

Example usage:

```python
from openai import OpenAI

model_id = "YOUR_MODEL_ID" # Replace with your model ID

client = OpenAI(
    api_key="YOUR-API-KEY",
    base_url=f"https://model-{model_id}.api.baseten.co/environments/production/sync/v1"
)

response = client.chat.completions.create(
    model="higgs-audio-v2-generation-3B-base",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Generate audio based on this description: upbeat electronic music"
                },
                {
                    "type": "audio_url",
                    "audio_url": {"url": "https://example.com/reference-audio.wav"}
                }
            ]
        }
    ],
    max_tokens=512,
    temperature=0.7
)

print(response.choices[0].message.content)
```

### Streaming Example

```python
response = client.chat.completions.create(
    model="higgs-audio-v2-generation-3B-base",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Create ambient background music"
                }
            ]
        }
    ],
    stream=True,
    max_tokens=512
)

for chunk in response:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
```

## Model Features

- **Audio Generation**: Generate high-quality audio content from text descriptions
- **Multimodal Input**: Accepts both text and audio inputs for context-aware generation
- **Flexible Length**: Supports up to 8192 tokens with configurable output length
- **Streaming Support**: Real-time streaming responses for interactive applications
- **Audio Tokenization**: Uses specialized audio tokenizer for optimal performance

## Configuration Options

The model is configured with several performance optimizations:

- **GPU Memory Utilization**: Set to 80% for optimal memory usage
- **Audio Limit**: Up to 50 audio inputs per prompt for complex audio generation tasks
- **Disabled MM Preprocessor Cache**: Ensures consistent performance across requests
- **Max Model Length**: 8192 tokens for extended context handling

## Support

If you have any questions or need assistance, please open an issue in this repository or contact our support team.

## License

Please refer to the original model's license at [bosonai/higgs-audio-v2-generation-3B-base](https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base) for usage terms and conditions.
