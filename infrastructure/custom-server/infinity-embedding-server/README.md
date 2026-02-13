# infinity-embedding-server

Deploy [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) using a custom server configuration on Baseten.

| Property | Value |
|----------|-------|
| Model | [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) |
| Task | Infrastructure / Custom server |
| Engine | Docker Server |
| GPU | L4 |

## Deploy

> **Note:** This model requires a HuggingFace access token. Set `hf_access_token` in your Baseten secrets before deploying.

```sh
truss push
```

## Invoke

```sh
curl -X POST https://model-<model_id>.api.baseten.co/embeddings \
  -H "Authorization: Api-Key YOUR_BASETEN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "BAAI/bge-small-en-v1.5", "input": ["What is the meaning of life?"]}'
```

## Configuration highlights

- Base image: `python:3.11-slim`
- Predict concurrency: **40**
- Environment variables: `INFINITY_MAX_CLIENT_BATCH_SIZE`, `INFINITY_QUEUE_SIZE`, `DO_NOT_TRACK`
