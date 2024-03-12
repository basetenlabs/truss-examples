# Whisper Streaming Truss

This is a truss for [Whisper Streaming](https://github.com/ufal/whisper_streaming). This truss allows you to stream the whisper transcription chunks as they get generated instead of waiting for the entire transcription to finish.

This whisper streaming model supports the following whisper models: `tiny.en`, `tiny`, `base.en`, `base`, `small.en`, `small`, `medium.en`, `medium`, `large-v1`, `large-v2`,`large-v3`, `large`.
You can specify the model you want to use inside the `config.yaml` file under the key `whisper_model` in the `model_metadata` section.

```yaml
model_metadata:
  whisper_model: medium
model_name: Whisper Streaming
```

## Deploying Whisper Streaming

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `whisper/whisper-streaming` as your working directory, you can deploy the model with:

```
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

## Whisper API documentation

The model accepts the following inputs:

- __audio__(required): The input audio file in the form of a base64 string.
- __chunk_size__(optional): The number of seconds of audio that will get transcribed in each chunk.



## Invoking the model

```python
import requests
import base64

def wav_to_base64(file_path):
    with open(file_path, "rb") as wav_file:
        binary_data = wav_file.read()
        base64_data = base64.b64encode(binary_data)
        base64_string = base64_data.decode("utf-8")
        return base64_string

resp = requests.post(
    "https://model-<model-id>.api.baseten.co/development/predict",
    headers = {"Authorization": "Api-Key BASETEN-API-KEY"},
    json={"audio": wav_to_base64("/path/to/wav/input_audio_file.wav")},
    stream=True
)

for content in resp.iter_content():
    print(content.decode("utf-8"), end="", flush=True)
```
