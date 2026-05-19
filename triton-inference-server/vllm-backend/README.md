# Triton + vLLM — OpenAI-compatible API on Baseten

Deploy [NVIDIA Triton](https://github.com/triton-inference-server/server) with the [vLLM backend](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/vllm_backend/README.html) and Triton's [OpenAI-compatible frontend](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client_guide/openai_readme.html) on Baseten.

The vLLM backend alone exposes Triton's `/v2/models/.../generate` API. This example runs `openai_frontend/main.py` so clients can use `/v1/chat/completions` and the OpenAI Python SDK.

## Layout

```
vllm-backend/
├── config.yaml
├── call.py
└── data/
    ├── start-openai.sh
    └── model_repository/
        └── llama-3.2-1b-instruct/    # OpenAI "model" name
            ├── config.pbtxt
            └── 1/model.json
```

`model.json` points the vLLM engine at `/models/llama` (BDN-mounted weights).

## Deploy

```bash
cd triton-inference-server/vllm-backend
truss push
```

Requires a Hugging Face token for `meta-llama/Llama-3.2-1B-Instruct` (`hf_access_token` secret).

## Inference

OpenAI chat completions (local, port 8000):

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "llama-3.2-1b-instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 64
  }'
```

On Baseten, POST the same body to `/environments/production/predict`, or use the sync route `/environments/production/sync/v1/chat/completions`.

## Client

```bash
pip install openai
export BASETEN_API_KEY=...
export BASETEN_MODEL_ID=<model_id>
python call.py
```

## Customize

| Goal | Where to change |
|------|-----------------|
| OpenAI model name | Rename `data/model_repository/<name>/` |
| HF weights path | `weights`, `model.json` (`model` field) |
| vLLM engine args | `1/model.json` |
| Triton / vLLM version | `base_image.image` (use `*-vllm-python-py3` tags) |

Also supports `/v1/completions` and `/v1/models`. See the [OpenAI frontend docs](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client_guide/openai_readme.html).
