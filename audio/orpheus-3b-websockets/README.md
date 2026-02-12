# Orpheus-3b Websockets

Deploy Orpheus-3b Websockets on Baseten using a TRT-LLM engine.

| Property | Value |
|----------|-------|
| Model | [baseten/orpheus-3b-0.1-ft](https://huggingface.co/baseten/orpheus-3b-0.1-ft) |
| Task | Audio |
| Engine | TRT-LLM |
| GPU | H100_40GB |
| Quantization | FP8 KV |
| Python | py39 |

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
  -d '{
  "max_tokens": 10000,
  "prompt": "In todays fast-paced world, finding balance between work and personal life is more important than ever. With the constant demands of technology, remote communication, ",
  "voice": "tara"
}'
```

## Configuration highlights

- Quantization: **fp8_kv**
- Max sequence length: **65,536**
- Plugin: **use_fp8_context_fmha**
- Environment variables: `ENABLE_EXECUTOR_API`
