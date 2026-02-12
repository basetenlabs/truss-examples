# Ultravox v0.2

Deploy Ultravox v0.2 for audio understanding on Baseten.

| Property | Value |
|----------|-------|
| Task | Audio understanding |
| Engine | vLLM |
| GPU | A100 |
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
  -d '{"model": "fixie-ai/ultravox-v0.2", "messages": [{"role": "user", "content": "Describe this audio."}]}'
```

## Configuration highlights

- Base image: `vshulman/vllm-openai-fixie:latest`
- Predict concurrency: **512**
- System packages: `python3.10-venv`
