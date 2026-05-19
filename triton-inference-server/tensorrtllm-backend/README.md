# Triton + TensorRT-LLM — OpenAI-compatible API on Baseten

Deploy [NVIDIA Triton](https://github.com/triton-inference-server/server) with the [TensorRT-LLM backend](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tensorrtllm_backend/README.html) and Triton's [OpenAI-compatible frontend](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client_guide/openai_readme.html) on Baseten.

Tokenizer weights come from [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) via BDN. You must supply a **Qwen3-8B TensorRT-LLM engine built for H100** at the engine `weights` source.

## Prerequisites

1. Build a Qwen3-8B TensorRT-LLM engine for H100 ([quick start](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tensorrtllm_backend/README.html)).
2. Upload engine artifacts via BDN and update `hf://YOUR_ORG/qwen3-8b-trtllm-engine-h100@main` in `config.yaml`.

## Layout

```
tensorrtllm-backend/
├── config.yaml
├── call.py
└── data/
    ├── start-openai.sh
    └── model_repository/
        ├── ensemble/           # OpenAI "model" name (default)
        ├── preprocessing/
        ├── postprocessing/
        └── tensorrt_llm/
```

## Deploy

```bash
cd triton-inference-server/tensorrtllm-backend
truss push
```

Runs on **H100**. `TRTLLM_ORCHESTRATOR=1` is set for tensor-parallel engines.

## Inference

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "ensemble",
    "messages": [{"role": "user", "content": "What is ML?"}],
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
