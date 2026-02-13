# Voxtral Small 24B 2507

Take in audio and text as input, generating text as usual

| Property | Value |
|----------|-------|
| Model | [mistralai/Voxtral-Small-24B-2507](https://huggingface.co/mistralai/Voxtral-Small-24B-2507) |
| Task | Infrastructure / Custom server |
| Engine | vLLM |
| GPU | H100 |
| OpenAI compatible | Yes |

## Deploy

> **Note:** This model requires a HuggingFace access token. Set `hf_access_token` in your Baseten secrets before deploying.

```sh
truss push
```

## Invoke

```sh
curl -X POST https://model-<model_id>.api.baseten.co/v1/chat/completions \
  -H "Authorization: Api-Key YOUR_BASETEN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "model": "voxtral-small",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What is the name of the famous bicycle race in France?"
        }
      ]
    }
  ]
}'
```

## Configuration highlights

- Base image: `vllm/vllm-openai:v0.10.0`
- Predict concurrency: **16**
