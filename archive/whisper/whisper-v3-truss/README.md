# Whisper v3 Truss

[Whisper](https://github.com/openai/whisper) is a open-source speech-to-text model by [OpenAI](https://openai.com/blog/whisper/) that transcribes audio in dozens of languages with remarkable accuracy.

Whisper v3 has the same architecture as the previous model but has made the following improvements:
- The large-v3 model is trained on 1 million hours of weakly labeled audio and 4 million hours of pseudolabeled audio collected using large-v2
- This version has a 10-20% lower rate of error compared to the previous version when benchmarked on `Common Voice 15` and `Fleurs` dataset


## Deploying Whisper

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `whisper-v3-truss` as your working directory, you can deploy the model with:

```
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

## Invoking Whisper

Once the model is deployed, you can invoke it with:

```sh
truss predict -d '{"url": "https://cdn.baseten.co/docs/production/Gettysburg.mp3"}'
```

You can also invoke your Whisper deployment via its REST API endpoint:

```bash
curl -X POST "https://model-{MODEL_ID}.api.baseten.co/development/predict" \
     -H "Content-Type: application/json" \
     -H 'Authorization: Api-Key {YOUR_API_KEY}' \
     -d '{"url": "https://cdn.baseten.co/docs/production/Gettysburg.mp3"}'
```

### Whisper API documentation

#### Input

This deployment of Whisper takes input as a JSON dictionary with the key `url` corresponding to a string of a URL pointing at an MP3 file. For example:

```json
{
    "url": "https://cdn.baseten.co/docs/production/Gettysburg.mp3"
}
```

#### Output

The model returns a fairly lengthy dictionary. For most uses, you'll be interested in the key `language` which specifies the detected language of the audio and `text` which contains the full transcription.

```json
{
    "language": "english",
    "segments": [
        {
        "start": 0,
        "end": 11.52,
        "text": "Four score and seven years ago our fathers brought forth upon this continent a new nation conceived in liberty and dedicated to the proposition that all men are created equal."
        }
    ],
    "text": "Four score and seven years ago our fathers brought forth upon this continent a new nation conceived in liberty and dedicated to the proposition that all men are created equal."
}
```
