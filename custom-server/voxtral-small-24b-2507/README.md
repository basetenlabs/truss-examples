# Voxtral Small 24B 2507 vLLM Truss

Voxtral Small is an enhancement of Mistral Small 3, incorporating state-of-the-art audio input capabilities while retaining best-in-class text performance. It excels at speech transcription, translation and audio understanding.

This is a [Truss](https://truss.baseten.co/) for Voxtral Small using the vLLM OpenAI Compatible server. This Truss bypasses the need for writing a `model.py` and instead runs `vllm serve` directly at startup and uses the HTTP endpoint provided by `vLLM` OpenAI Compatible Server to directly serve requests.

## Deployment

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples.git
cd custom-server/voxtral-small-24b-2507
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`
3. Retrieve your Hugging Face token from the [settings](https://huggingface.co/settings/tokens).
4. Set your Hugging Face token as a Baseten secret [here](https://app.baseten.co/settings/secrets) with the key `hf_access_key`. Note that you will _not_ be able to successfully deploy Voxtral Small without doing this.

With `voxtral-small-24b-2507` as your working directory, you can deploy the model with:

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
from mistral_common.protocol.instruct.messages import (
    TextChunk,
    AudioChunk,
    UserMessage,
    AssistantMessage,
    RawAudio,
)
from mistral_common.audio import Audio
from huggingface_hub import hf_hub_download

from openai import OpenAI

model_id = "12345678"

client = OpenAI(
    api_key="YOUR_API_KEY",
    base_url=f"https://model-{model_id}.api.baseten.co/{deploy_env}/sync/v1"
)

models = client.models.list()
model = models.data[0].id

obama_file = hf_hub_download(
    "patrickvonplaten/audio_samples", "obama.mp3", repo_type="dataset"
)
bcn_file = hf_hub_download(
    "patrickvonplaten/audio_samples", "bcn_weather.mp3", repo_type="dataset"
)

def file_to_chunk(file: str) -> AudioChunk:
    audio = Audio.from_file(file, strict=False)
    return AudioChunk.from_audio(audio)

text_chunk = TextChunk(
    text="Which speaker is more inspiring? Why? How are they different from each other? Answer in French."
)
user_msg = UserMessage(
    content=[file_to_chunk(obama_file), file_to_chunk(bcn_file), text_chunk]
).to_openai()

print(30 * "=" + "USER 1" + 30 * "=")
print(text_chunk.text)
print("\n\n")

response = client.chat.completions.create(
    model=model,
    messages=[user_msg],
    temperature=0.2,
    top_p=0.95,
)
content = response.choices[0].message.content

print(30 * "=" + "BOT 1" + 30 * "=")
print(content)
print("\n\n")

messages = [
    user_msg,
    AssistantMessage(content=content).to_openai(),
    UserMessage(
        content="Ok, now please summarize the content of the first audio."
    ).to_openai(),
]
print(30 * "=" + "USER 2" + 30 * "=")
print(messages[-1]["content"])
print("\n\n")

response = client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=0.2,
    top_p=0.95,
)
content = response.choices[0].message.content
print(30 * "=" + "BOT 2" + 30 * "=")
print(content)

```

## Support

If you have any questions or need assistance, please open an issue in this repository or contact our support team.
