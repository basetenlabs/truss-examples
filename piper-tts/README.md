# Piper TTS Truss

This repository packages [Piper-TTS](https://github.com/rhasspy/piper) as a [Truss](https://truss.baseten.co/).

Piper TTS is a generative audio model for text-to-speech generation. This model can be trained on various voices to create realistic output audio. This model has very low latency and is optimized to run on a Raspberry Pi!

## Deploying Piper TTS

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd piper-tts
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `piper-tts` as your working directory, you can deploy the model with:

```sh
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).


## Hardware Requirements
This model does not need a GPU to run. However, the configuration provided with this Truss does have a GPU enabled for extra performance.

## Invoking the model

Here are the following inputs for the model:
1. `text`: The text that needs to be converted into speech

Here is an example of how to invoke this model:

```python
import base64
import requests

def base64_to_wav(base64_string, output_file_path):
  binary_data = base64.b64decode(base64_string)
  with open(output_file_path, "wb") as wav_file:
    wav_file.write(binary_data)

text = "Listen up, people. Life's a wild ride, and sometimes you gotta grab it by the horns and steer it where you want to go. You can't just sit around waiting for things to happen – you gotta make 'em happen. Yeah, it's gonna get tough, but that's when you dig deep, find that inner badass, and come out swinging. Remember, success ain't handed to you on a silver platter; you gotta snatch it like it owes you money. So, lace up your boots, square those shoulders, and let the world know that you're here to play, and you're playing for keeps"
data = {"text": text}
headers = {"Authorization": f"Api-Key <BASETEN-API-KEY>"}
res = requests.post("https://model-<model-id>.api.baseten.co/development/predict", headers=headers, json=data)
res = res.json()
output = base64_to_wav(res.get('output'), "piper-tts-output.wav")
print(res)
```

The output of the model is a base64 string, so you can convert it to a wav file using the `base64_to_wav` function.

Here is the output from the model using the input above:
