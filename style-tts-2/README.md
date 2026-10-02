# Style TTS 2 Truss

This repository packages [Style-TTS-2](https://github.com/yl4579/StyleTTS2) as a [Truss](https://truss.baseten.co/).

Style TTS is a text to speech model designed to achieve human-level TTS synthesis.

## Deploying Style TTS 2

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd style-tts-2
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `style-tts-2` as your working directory, you can deploy the model with:

```sh
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).


## Invoking the model

Here are the following inputs for the model:
1. `text`: The text that needs to be converted into speech
2. `reference_audio` (optional): This is an audio file is used for voice cloning in the output audio. The `reference_audio` must be a base64 string.

Here is an example of how to invoke this model:

```python
import base64
import requests
import os

def base64_to_wav(base64_string, output_file_path):
  binary_data = base64.b64decode(base64_string)
  with open(output_file_path, "wb") as wav_file:
    wav_file.write(binary_data)

text = "StyleTTS 2 is a text-to-speech model that leverages style diffusion and adversarial training with large speech language models to achieve human-level text-to-speech synthesis."
data = {"text": text}
headers = {"Authorization": f"Api-Key <BASETEN-API-KEY>"}
res = requests.post("https://model-<model-id>.api.baseten.co/development/predict", headers=headers, json=data)
res = res.json()
output = base64_to_wav(res.get('output'), "style-tts-output.wav")
os.system("open style-tts-output.wav")
```

The output of the model is a base64 string, so you can convert it to a wav file using the `base64_to_wav` function.

Here is the output from the model using the input above:

https://github.com/htrivedi99/truss-examples/assets/15642666/08361f8e-8951-4a01-a5f1-459c0805b31f
