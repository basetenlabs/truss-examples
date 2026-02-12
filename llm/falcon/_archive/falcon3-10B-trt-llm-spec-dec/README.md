# Falcon 3 10B Instruct

Deploy [tiiuae/Falcon3-10B-Instruct](https://huggingface.co/tiiuae/Falcon3-10B-Instruct) for text generation using a TRT-LLM engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [tiiuae/Falcon3-10B-Instruct](https://huggingface.co/tiiuae/Falcon3-10B-Instruct) |
| Task | Text generation |
| Engine | TRT-LLM |
| GPU | H100 |
| Quantization | NO QUANT |

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
  "frequency_penalty": 1,
  "max_tokens": 512,
  "messages": [
    {
      "content": "You are a knowledgable, engaging, biology teacher.",
      "role": "system"
    },
    {
      "content": "What makes falcons effective hunters?",
      "role": "user"
    }
  ],
  "stream": true,
  "temperature": 0.6
}'
```

## Configuration highlights

- Quantization: **no_quant**
- Speculative decoding: **DRAFT_TOKENS_EXTERNAL**
- Max sequence length: **8,192**
- Chunked context: **enabled**
- Plugin: **paged_kv_cache**
- Plugin: **use_paged_context_fmha**
- Streaming: **enabled**
