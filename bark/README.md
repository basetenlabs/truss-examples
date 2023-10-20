# Bark Truss

This repository packages [Bark](https://github.com/suno-ai/bark) as a [Truss](https://truss.baseten.co/).

Bark is a generative audio model for text-to-speech generation.

## Deploying Bark

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd bark-truss
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `bark-truss` as your working directory, you can deploy the model with:

```sh
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).
## Invoking Bark

Bark takes a string as its input and returns a Base64-encoded WAV audio as output. Bark currently works best for strings resulting in up to 12 seconds of audio, or approximately 20-25 words in English.

Here's an example invocation that decodes and saves the output to a file:

```sh
truss predict -d '"Two elevator mechanics discuss everything they hate about escalators"' | python process.py
```

With `process.py` as follows:

```python

import base64
import sys

b64_audio = sys.stdin.read()

wav_file = open("bark_output.wav", "wb")
decode_string = base64.b64decode(b64_audio)
wav_file.write(decode_string)
```
