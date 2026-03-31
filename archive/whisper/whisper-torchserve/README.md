# Whisper Torchserve

This truss allows you to run a whisper model using [torchserve](https://pytorch.org/serve/) as the backend on truss.


## Deployment

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `whisper/whisper-torchserve` as your working directory, you can deploy the model with:

```
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

## Model Inputs

The model takes in one input:
- __audio__: An audio file as a base64 string

## Few thing to note
Torchserve requires a compiled `.mar` file in order to serve the model. Here is a [README](https://github.com/pytorch/serve/blob/master/model-archiver/README.md) providing a brief explanation for generating this file. Once the `.mar` file is generated it needs to get placed in the `data/model_store` directory. Also in the `data/` directory is a configuration file for torchserve called `config.properties`. That file looks something like this:

```
inference_address=http://0.0.0.0:8888
batch_size=4
ipex_enable=true
async_logging=true

models={\
  "whisper_base": {\
    "1.0": {\
        "defaultVersion": true,\
        "marName": "whisper_base.mar",\
        "minWorkers": 1,\
        "maxWorkers": 2,\
        "batchSize": 4,\
        "maxBatchDelay": 500,\
        "responseTimeout": 24\
    }\
  }\
}
```

Here you can specify the `batchSize` as well as the name of your mar file using `marName`. When torchserve starts, it will looks for the mar file inside the `data/model_store` directory with the `marName` defined above.

## Invoking the model

Here is an example in Python:

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
    headers={"Authorization": "Api-Key BASETEN-API-KEY"},
    json={"audio": wav_to_base64("/path/to/audio-file/60-sec.wav")},
)

print(resp.json())
```

Here is a sample output:

```json
{"output": "Let me make it clear. His conduct is unacceptable. He's unfit. And be careful of what you're gonna get. He doesn't care for the American people. It's Donald Trump first. This is what I want people to understand. These people have... I mean, she has no idea what the hell the names of those provinces are, but she wants to send our sons and daughters and our troops and our military equipment to go fight it. Look at the blank expression. She doesn't know the names of the provinces. You do this at every debate. You say, no, don't interrupt me. I didn't interrupt you."}
```
