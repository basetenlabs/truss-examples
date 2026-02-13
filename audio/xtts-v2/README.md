# XTTS V2

Deploy XTTS V2 for text-to-speech on Baseten.

| Property | Value |
|----------|-------|
| Task | Text-to-speech |
| Engine | Custom (Truss) |
| GPU | T4 |
| Python | py310 |

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
  "language": "en",
  "speaker_voice": "Claribel Dervla",
  "text": "Kurt watched the incoming Pelicans. The blocky jet-powered craft were so distant they were only specks against the setting sun. He hit the magnification on his faceplate and saw lines of fire tracing their reentry vectors. They would touch down in three minutes."
}'
```

## Configuration highlights

- Environment variables: `COQUI_TOS_AGREED`
