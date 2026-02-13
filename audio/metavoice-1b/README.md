# MetaVoice 1B

MetaVoice is a transformer-based model for TTS

| Property | Value |
|----------|-------|
| Model | [metavoiceio/metavoice-1B-v0.1](https://huggingface.co/metavoiceio/metavoice-1B-v0.1) |
| Task | Text-to-speech |
| Engine | Custom (Truss) |
| GPU | A10G |
| Python | py311 |

## Deploy

> **Note:** This model requires a HuggingFace access token. Set `hf_access_token` in your Baseten secrets before deploying.

```sh
truss push
```

## Invoke

```sh
curl -X POST https://model-<model_id>.api.baseten.co/predict \
  -H "Authorization: Api-Key YOUR_BASETEN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '"text to speech models are cool"'
```

## Configuration highlights

- System packages: `ffmpeg`
