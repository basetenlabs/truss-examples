# Bark Truss

This repository packages [Bark](https://github.com/suno-ai/bark) as a [Truss](https://truss.baseten.co/).

Bark is a generative audio model for text-to-speech generation.

## Deploying Bark

Bark runs well on an NVIDIA A10, with an inference time of 3-5 seconds depending on the length of the prompt.

Before deployment:

1. Make sure you have a Baseten account and API key. You can sign up for a Baseten account [here](https://app.baseten.co/signup).
2. Install Truss and the Baseten Python client: `pip install --upgrade baseten truss`
3. Authenticate your development environment with `baseten login`

Deploying the Truss is easy; simply load it and push from a Python script:

```python
import baseten
import truss

bark_truss = truss.load('.')
baseten.deploy(bark_truss, model_name="Bark")
```

## Invoking Bark

Bark takes a string as its input and returns a Base64-encoded WAV audio as output. Bark currently works best for strings resulting in up to 12 seconds of audio, or approximately 20-25 words in English.

Here's an example invocation that decodes and saves the output to a file:

```python
import baseten
import base64

model_input = "Two elevator mechanics discuss everything they hate about escalators"

bark = baseten.deployed_model_version_id('MODEL_VERSION_ID')
b64_audio = bark.predict(model_input)

wav_file = open("bark_output.wav", "wb")
decode_string = base64.b64decode(b64_audio)
wav_file.write(decode_string)
```
