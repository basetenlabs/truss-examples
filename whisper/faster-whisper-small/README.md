# Whisper Small

This is a [Truss](https://truss.baseten.co/) for Whisper Small. This README will walk you through how to deploy this Truss on Baseten to get your own instance of Whisper Small up and running.

## Faster Whisper Small Implementation

This implementation of Whisper Small uses [Faster Whisper](https://github.com/SYSTRAN/faster-whisper/tree/master), which is up to 4x faster than openai/whisper for the same accuracy while using less memory.

## Deployment

You can deploy this model in just a few clicks from our [model library](), or deploy the Truss, which we'll describe here.

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd whisper/faster-whisper-small
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `whisper/faster-whisper-small` as your working directory, you can deploy the model with:

```sh
truss push --trusted
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

Once your Truss is deployed, you can start using Whisper Small for inference! Navigate to the [Baseten UI](https://app.baseten.co/models) to watch the model build and deploy and invoke it via the REST API for tasks like transcription.

## Example usage

Here's a sample script which loads a sample .mp3 file for transcription with Whisper Small:

```python
import requests
import os

# Replace the empty string with your model id below
model_id = ""

# We recommend storing your API key as an environment variable
baseten_api_key = os.environ["BASETEN_API_KEY"]

data = {
  "url": "https://cdn.baseten.co/docs/production/Gettysburg.mp3"
}

# Call model endpoint
res = requests.post(
    f"https://model-{model_id}.api.baseten.co/production/predict",
    headers={"Authorization": f"Api-Key {baseten_api_key}"},
    json=data
)

# Print the output of the model
print(res.json())
```

Here is the model output:
```json
{'language': 'en', 'language_probability': 0.99072265625, 'duration': 11.52, 'segments': [{'text': ' Four score and seven years ago, our fathers brought forth upon this continent a new nation', 'start': 0.0, 'end': 6.5200000000000005}, {'text': ' conceived in liberty and dedicated to the proposition that all men are created equal.', 'start': 6.5200000000000005, 'end': 11.0}]}
```