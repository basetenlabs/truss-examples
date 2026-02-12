# kokoro

Deploy kokoro for text-to-speech on Baseten.

| Property | Value |
|----------|-------|
| Task | Text-to-speech |
| Engine | Custom (Truss) |
| GPU | T4 |
| Python | py311 |

## Deploy

```sh
truss push
```

## Invoke

```sh
curl -X POST https://model-<model_id>.api.baseten.co/predict \
  -H "Authorization: Api-Key YOUR_BASETEN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "text": "Kokoro is a frontier TTS model for its size of 82 million parameters (text in/audio out). On 25 Dec 2024, Kokoro v0.19 weights were permissively released in full fp32 precision under an Apache 2.0 license. As of 2 Jan 2025, 10 unique Voicepacks have been released, and a .onnx version of v0.19 is available.In the weeks leading up to its release, Kokoro v0.19 was the #1\ud83e\udd47 ranked model in TTS Spaces Arena. Kokoro had achieved higher Elo in this single-voice Arena setting over other models, using fewer parameters and less data. Kokoro's ability to top this Elo ladder suggests that the scaling law (Elo vs compute/data/params) for traditional TTS models might have a steeper slope than previously expected.",
  "voice": "af",
  "speed": 1.0
}'
```

## Configuration highlights

- Predict concurrency: **1**
- System packages: `espeak-ng`
