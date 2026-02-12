# text-embeddings-inference trussless

Deploy [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) for generating text embeddings using a TEI (HuggingFace) engine on Baseten.

| Property | Value |
|----------|-------|
| Model | [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) |
| Task | Embeddings |
| Engine | TEI (HuggingFace) |
| GPU | L4 |

## Deploy

> **Note:** This model requires a HuggingFace access token. Set `hf_access_token` in your Baseten secrets before deploying.

```sh
truss push
```

## Invoke

```sh
curl -X POST https://model-<model_id>.api.baseten.co/v1/embeddings \
  -H "Authorization: Api-Key YOUR_BASETEN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": "What is deep learning?", "model": "BAAI/bge-base-en-v1.5"}'
```

## Configuration highlights

- Base image: `baseten/text-embeddings-inference-mirror:89-1.6`
- Predict concurrency: **40**
- Server port: **7997**
