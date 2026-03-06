# Qwen3 TTS Base (Voice Cloning)

Voices are cached on a per-replica basis for the time being with a fast-follow to have caching take place across replicas layer so all replicas can share computed speaker embeddings.

## Usage

First, send an initial cloning request with a voice name for referring to the clone

```
python call.py --text "Hey my name is Alex, I'm your helpful assistant" --ref-audio alex.m4a --ref-text transcript.txt --voice-name Alex
```

Once, this initial request has been completed the clone will be stored under the specified voice name and reference audio/text no longer need to be passed.

```
python call.py --text "Hey my name is Alex, I'm your helpful assistant" --voice-name Alex
```
