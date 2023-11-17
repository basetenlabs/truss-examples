# XTTS V2 Truss

This repository packages [TTS](https://github.com/coqui-ai/TTS) as a [Truss](https://truss.baseten.co/).

TTS is a generative audio model for text-to-speech generation. This model takes in text and a speaker's voice as input and converts the text to speech in the voice of the speaker.

## Deploying XTTS

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd xtts-v2-truss
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `xtts-v2-truss` as your working directory, you can deploy the model with:

```sh
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

## Invoking the model

Here are the following inputs for the model:
1. `text`: The text that needs to be converted into speech
2. `speaker_voice`: A short audio clip of a voice in the format of a base64 string
3. `language`: Abbreviation of the supported languages for TTS

Here is an example of how to invoke this model:

```python

import base64
import sys

def wav_to_base64(file_path):
  with open(file_path, "rb") as wav_file:
    binary_data = wav_file.read()
    base64_data = base64.b64encode(binary_data)
    base64_string = base64_data.decode("utf-8")
    return base64_string

def base64_to_wav(base64_string, output_file_path):
  binary_data = base64.b64decode(base64_string)
  with open(output_file_path, "wb") as wav_file:
    wav_file.write(binary_data)

voice = wav_to_base64("/path/to/wav/file/samuel_jackson_voice.wav")
text = "Listen up, people. Life's a wild ride, and sometimes you gotta grab it by the horns and steer it where you want to go. You can't just sit around waiting for things to happen – you gotta make 'em happen. Yeah, it's gonna get tough, but that's when you dig deep, find that inner badass, and come out swinging. Remember, success ain't handed to you on a silver platter; you gotta snatch it like it owes you money. So, lace up your boots, square those shoulders, and let the world know that you're here to play, and you're playing for keeps"
data = {"text": text, "speaker_voice": voice, "language": "en"}
res = requests.post("https://model-<model-id>.api.baseten.co/development/predict", headers=headers, json=data)
res = res.json()
output = base64_to_wav(res.get('output'), "test_output.wav")
```

The output of the model is a base64 string as well, so you can convert it to a wav file using the `base64_to_wav` function.

Here is the input file for Samuel Jackson's voice:
![speaker voice](samuel_jackson_voice.mp4)
