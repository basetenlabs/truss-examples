# Ultravox v0.4

Take in audio and text as input, generating text as usual

| Property | Value |
|----------|-------|
| Model | [fixie-ai/ultravox-v0.4](https://huggingface.co/fixie-ai/ultravox-v0.4) |
| Task | Infrastructure / Custom server |
| Engine | vLLM |
| GPU | H100_40GB |
| OpenAI compatible | Yes |

## Deploy

```sh
truss push
```

## Invoke

```sh
curl -X POST https://model-<model_id>.api.baseten.co/v1/chat/completions \
  -H "Authorization: Api-Key YOUR_BASETEN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "model": "ultravox",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What is Lydia like?"
        },
        {
          "type": "audio_url",
          "audio_url": {
            "url": "https://baseten-public.s3.us-west-2.amazonaws.com/fred-audio-tests/real.mp3"
          }
        }
      ]
    }
  ]
}'
```

## Configuration highlights

- Base image: `vllm/vllm-openai:v0.9.2`
- Predict concurrency: **16**
