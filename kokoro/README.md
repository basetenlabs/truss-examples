Kokoro is a frontier TTS model for its size of 82 million parameters (text in/audio out).
API:
```bash
request:
{"text": "Hello", "voice": "af", "speed": 1.0}

text: str = defaults to "Hi, I'm kokoro"
voice: str = defaults to "af", available options: "af", "af_bella", "af_sarah", "am_adam", "am_michael", "bf_emma", "bf_isabella", "bm_george", "bm_lewis", "af_nicole", "af_sky"
speed: float = defaults to 1.0. The speed of the audio generated

reponse:
{"base64": "base64 encoded bytestring"}
```
