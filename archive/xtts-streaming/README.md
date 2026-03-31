# XTTS Streaming

This repository packages [TTS](https://github.com/coqui-ai/TTS) as a [Truss](https://truss.baseten.co/) but with streaming.

TTS is a generative audio model for text-to-speech generation. This model takes in text and a speaker's voice as input and converts the text to speech in the voice of the speaker.

## Deploying XTTS

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd xtts-streaming
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
2. `language`: Language for the text
3. `chunk_size`: Integer size of each chunk being streamed

Here are two examples of streaming the audio. This first example write all of the streamed chunks to an audio file.

```python
import wave
import requests

channels = 1  # mono=1, stereo=2
sampwidth = 2  # Sample width in bytes, typical values: 2 for 16-bit audio, 1 for 8-bit audio
framerate = 24000  # Sampling rate, in samples per second (Hz)


resp = requests.post(
    "https://model-<model-id>.api.baseten.co/development/predict",
    headers={"Authorization": "Api-Key BASETEN-API-KEY"},
    json={"text": "Kurt watched the incoming Pelicans. The blocky jet-powered craft were so distant they were only specks against the setting sun. He hit the magnification on his faceplate and saw lines of fire tracing their reentry vectors. They would touch down in three minutes."},
    stream=True
)

with wave.open("dat2-wav.wav", 'wb') as wav_file:
    wav_file.setnchannels(channels)
    wav_file.setsampwidth(sampwidth)
    wav_file.setframerate(framerate)

    # Iterate through streamed content and write audio chunks directly
    for chunk in resp.iter_content(chunk_size=None):  # Use server's chunk size
        if chunk:
            wav_file.writeframes(chunk)
```

If you want to stream the audio directly as it gets generated here is another option:

```python
import pyaudio

FORMAT = pyaudio.paInt16  # Audio format (e.g., 16-bit PCM)
CHANNELS = 1              # Number of audio channels
RATE = 24000              # Sample rate

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open a stream for audio playback
stream = p.open(format=p.get_format_from_width(2), channels=CHANNELS, rate=RATE, output=True)

# Make a streaming HTTP request to the server
original_text = "Kurt watched the incoming Pelicans. The blocky jet-powered craft were so distant they were only specks against the setting sun. He hit the magnification on his faceplate and saw lines of fire tracing their reentry vectors. They would touch down in three minutes."


resp = requests.post(
    "https://model-<model-id>.api.baseten.co/development/predict",
    headers={"Authorization": "Api-Key BASETEN-API-KEY"},
    json={"text": "Kurt watched the incoming Pelicans. The blocky jet-powered craft were so distant they were only specks against the setting sun. He hit the magnification on his faceplate and saw lines of fire tracing their reentry vectors. They would touch down in three minutes."},
    stream=True
)

# Create a buffer to hold multiple chunks
buffer = b''
buffer_size_threshold = 2**20

# Stream and play the audio data as it's received
for chunk in resp.iter_content(chunk_size=4096):
    if chunk:
        buffer += chunk
        if len(buffer) >= buffer_size_threshold:
            print(f"Writing buffer of size: {len(buffer)}")
            stream.write(buffer)
            buffer = b''  # Clear the buffer
        # stream.write(chunk)

if buffer:
    print(f"Writing final buffer of size: {len(buffer)}")
    stream.write(buffer)

# Close and terminate the stream and PyAudio
stream.stop_stream()
stream.close()
p.terminate()
```
