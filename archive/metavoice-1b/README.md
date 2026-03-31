# MetaVoice 1B Truss
[MetaVoice-1B](https://github.com/metavoiceio/metavoice-src) is an Apache-licensed, 1.2B parameter base model trained on 100K hours of speech for TTS (text-to-speech).

This model is packaged using [Truss](https://trussml.com), the simplest way to serve AI/ML models in production.

## Deploy MetaVoice 1B
First, clone this repository:

```
git clone https://github.com/basetenlabs/truss-examples/
cd metavoice-1b
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `metavoice-1b` as your working directory, you can deploy the model with:

```
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

Once your Truss is deployed, you can start using MetaVoice through the Baseten platform! Navigate to the Baseten UI to watch the model build and deploy and invoke it via the REST API.


## Invoking MetaVoice

To use MetaVoice 1B, follow this command pattern, keeping in mind the 220-character limit for input text:

```sh
truss predict -d '{"text": "Your input text here"}' | python process.py
```

### Understanding process.py
The process.py script is essential for handling the output from MetaVoice 1B. It reads the base64 encoded audio from the standard input, decodes it, and saves it as a WAV file. Here's a brief overview of how it works:

```python
import base64
import sys

b64_audio = sys.stdin.read()
b64_audio = b64_audio.split('"')[1]  # Extracting the base64 string

wav_file = open("output.wav", "wb")
decode_string = base64.b64decode(b64_audio)
wav_file.write(decode_string)
```

## Notes
- `flash_attn` requires installation with `--no-build-isolation`. As this isn't supported, installing the wheel directly seems to work (see `config.yaml`).

## To Dos
- Add support for passing additional reference voices to use
