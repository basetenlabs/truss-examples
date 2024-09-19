# Audio Transcription Chain

This chain can transcribe any media files that are supported by the
[ffmpeg](https://ffmpeg.org/) library and that are hosted under a URL that
supports range downloads. Very large files are supported in near-constant time using chunking.

More details are described in the
[guide](https://docs.baseten.co/chains/examples/audio-transcription) (note that the docs will be moved soon and the link might need to be updated).

To support transcribing audio that is longer than 30 seconds, the first chainlet will chunk the clip into segments less than 30 seconds using [Silero VAD](https://github.com/snakers4/silero-vad). These chunks are then batch processed by the second chainlet using OpenAI's [Whisper V2 Large](https://huggingface.co/openai/whisper-large-v2) model.

## Deploying with `truss`

To deploy this chain:

```bash
truss chains deploy vad_transcribe.py
```

## Invoking

The chain accepts either a url to a audio file, or a base 64 string representing an audio clip:

```
curl -X POST "https://chain-{CHAIN_ID}.api.baseten.co/production/run_remote" \
     -H "Content-Type: application/json" \
     -H 'Authorization: Api-Key {YOUR_API_KEY}' \
     -d '{"whisper_input": {"audio": {"url": "https://cdn.baseten.co/docs/production/Gettysburg.mp3"}}}'
```

## Output

The model returns a dictionary which contains the start and end timestamps along with the transcript of what is said in between each timestamp. The model also outputs the language of the audio.

```
{
  "segments": [
    {
      "start_time_sec": 0.268,
      "end_time_sec": 6.508,
      "text": "\"Four score and seven years ago, our fathers brought forth upon this continent a new nation"
    },
    {
      "start_time_sec": 6.508,
      "end_time_sec": 11,
      "text": "conceived in liberty and dedicated to the proposition that all men are created equal.\""
    }
  ],
  "language_code": "en"
}
```


