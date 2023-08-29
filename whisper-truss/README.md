[![Deploy to Baseten](https://user-images.githubusercontent.com/2389286/236301770-16f46d4f-4e23-4db5-9462-f578ec31e751.svg)](https://app.baseten.co/explore/whisper)

# Whisper Truss

[Whisper](https://github.com/openai/whisper) is a speech-to-text model by [OpenAI](https://openai.com/blog/whisper/) that transcribes audio in dozens of languages with remarkable accuracy. It is open-source under the [MIT license](https://github.com/openai/whisper/blob/main/LICENSE) and hosted on Baseten as a pre-trained model. Read the [Whisper model card](https://github.com/openai/whisper/blob/main/model-card.md) for more details.

Whisper's leap in transcription quality unlocks tons of compelling use cases, including:

* Moderating audio content
* Auditing call center logs
* Automatically generating video subtitles
* Improving podcast SEO with transcripts

## Deploying Whisper

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `whisper-truss` as your working directory, you can deploy the model with:

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
curl -X POST "https://app.baseten.co/models/{MODEL_ID}/predict" \
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
        "end": 6.5200000000000005,
        "text": " Four score and seven years ago, our fathers brought forth upon this continent a new nation"
        },
        {
        "start": 6.52,
        "end": 21.6,
        "text": " conceived in liberty and dedicated to the proposition that all men are created equal."
        }
    ],
    "text": " Four score and seven years ago, our fathers brought forth upon this continent..."
}
```
