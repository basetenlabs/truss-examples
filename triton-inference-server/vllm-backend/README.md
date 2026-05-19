# Triton + vLLM — OpenAI-compatible API on Baseten

Deploy [NVIDIA Triton](https://github.com/triton-inference-server/server) with the [vLLM backend](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/vllm_backend/README.html) and Triton's [OpenAI-compatible frontend](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client_guide/openai_readme.html) on Baseten.

Serves [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) on **H100** via BDN.

## Layout

```
vllm-backend/
├── config.yaml
├── call.py
└── data/
    ├── start-openai.sh
    └── model_repository/
        └── qwen3-8b/              # OpenAI "model" name
            ├── config.pbtxt
            └── 1/model.json
```

`model.json` points the vLLM engine at `/models/qwen` (BDN-mounted `Qwen/Qwen3-8B`).

## Deploy

```bash
cd triton-inference-server/vllm-backend
truss push
```

## Inference

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-8b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 64
  }'
```

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
| HF weights | `weights` (`hf://Qwen/Qwen3-8B`), `model.json` |
| vLLM engine args | `1/model.json` |
| GPU | `resources.accelerator` |
