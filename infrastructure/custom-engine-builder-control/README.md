# Briton-suffix-fanout-qwen3-8B

Deploy [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) using a custom server configuration on Baseten.

| Property | Value |
|----------|-------|
| Model | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) |
| Task | Infrastructure / Custom server |
| Engine | TRT-LLM |
| GPU | H100 |
| Quantization | FP8 KV |
| OpenAI compatible | Yes |
| Python | py39 |

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
  "max_tokens": 100,
  "messages": [
    {
      "content": "You are a helpful assistant. To each math question, e.g. <Whats 1+1>, respond with all parts the question, '=', and answer.",
      "role": "system"
    }
  ],
  "stream": false,
  "suffix_messages": [
    [
      {
        "role": "system",
        "content": "Whats 1+1"
      }
    ],
    [
      {
        "role": "system",
        "content": "Whats 2+2"
      }
    ]
  ],
  "temperature": 0.5,
  "chat_template_kwargs": {
    "enable_thinking": false
  }
}'
```

## Configuration highlights

- Quantization: **fp8_kv**
- Speculative decoding: **LOOKAHEAD_DECODING**
- Max sequence length: **32,768**
- Plugin: **use_fp8_context_fmha**
