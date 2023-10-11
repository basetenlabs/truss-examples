# WhisperX Truss

[WhisperX](https://github.com/m-bain/whisperX) is a model built on top of Whisper that provides fast speech recognition with word-level timestamps. 


## Deploying WhisperX

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

Next, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd whisperx-truss
```

With `whisperx-truss` as your working directory, you can deploy the model with:

```
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

## Hardware notes

This whisperX model comes in various sizes: ["small", "medium", "large-v2"].
The large-v2 model can be easily run on the T4 GPU. 

## API route: `predict`

The `predict` route is the primary method used for audio transcription. 

- __audio_file__: An MP3 audio file. This file must be accessible over the internet as files from local storage are not accessible.

## Invoking WhisperX

Once the model is deployed, you can invoke it with:

```sh
truss predict -d '{"audio_file": "https://cdn.baseten.co/docs/production/Gettysburg.mp3"}'
```

You can also invoke your Whisper deployment via its REST API endpoint:

```bash
curl -X POST "https://app.baseten.co/models/{MODEL_ID}/predict" \
     -H "Content-Type: application/json" \
     -H 'Authorization: Api-Key {YOUR_API_KEY}' \
     -d '{"audio_file": "https://cdn.baseten.co/docs/production/Gettysburg.mp3"}'
```

## Output

The model returns a dictionary which contains the start and end timestamps along with the transcript of what is said in between each timestamp. 

```json
{
    "model_output": 
    [
        {
            "end": 10.742, 
            "start": 0.765, 
            "text": "Four score and seven years ago, our fathers brought forth upon this continent, a new nation conceived in liberty and dedicated to the proposition that all men are created equal."
        }
    ]
}
```
