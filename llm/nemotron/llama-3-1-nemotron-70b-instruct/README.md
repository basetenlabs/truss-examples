# Llama-3.1-Nemotron-70B-Instruct

Deploy [nvidia/Llama-3.1-Nemotron-70B-Instruct-HF](https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Instruct-HF) for text generation using a TRT-LLM engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [nvidia/Llama-3.1-Nemotron-70B-Instruct-HF](https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Instruct-HF) |
| Task | Text generation |
| Engine | TRT-LLM |
| GPU | H100:2 |
| Quantization | FP8 KV |
| Python | py39 |

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
  "messages": [
    {
      "role": "user",
      "content": "How many r in strawberry?"
    }
  ],
  "stream": true,
  "max_tokens": 512,
  "temperature": 0.6
}'
```

## Configuration highlights

- Quantization: **fp8_kv**
- Tensor parallelism: **2** GPUs
- Max sequence length: **131,072**
- Chunked context: **enabled**
- Batch scheduler policy: **max_utilization**
- Plugin: **use_paged_context_fmha**
- Plugin: **use_fp8_context_fmha**
- Plugin: **paged_kv_cache**
- Streaming: **enabled**
