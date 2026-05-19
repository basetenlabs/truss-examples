# Triton + TensorRT-LLM (PyTorch / LLMAPI) — OpenAI API on Baseten

Deploy [NVIDIA Triton](https://github.com/triton-inference-server/server) with the TensorRT-LLM [**LLMAPI PyTorch backend**](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tensorrtllm_backend/docs/llmapi.html) and Triton's [OpenAI-compatible frontend](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client_guide/openai_readme.html) on Baseten.

This path uses `backend: pytorch` in `model.yaml` and loads [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) from BDN at runtime. **No pre-built TensorRT engine is required** (unlike the `inflight_batcher_llm` C++ backend).

## Layout

```
tensorrtllm-backend/
├── config.yaml
├── call.py
└── data/
    ├── start-openai.sh
    └── model_repository/
        └── tensorrt_llm/
            ├── config.pbtxt
            └── 1/
                ├── model.yaml    # LLMAPI config (model path, backend: pytorch)
                └── model.py      # Triton Python backend entrypoint
```

## Deploy

```bash
cd triton-inference-server/tensorrtllm-backend
truss push
```

Runs on **H100**. Weights mount at `/models/qwen`; `model.yaml` sets `model: /models/qwen`.

## Inference

OpenAI chat completions — use Triton model name `tensorrt_llm`:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "tensorrt_llm",
    "messages": [{"role": "user", "content": "What is ML?"}],
    "max_completion_tokens": 64
  }'
```

The LLMAPI `config.pbtxt` also accepts OpenAI-frontend aliases (`temperature`, `max_tokens`, `stream`, etc.) via `helpers.py`.

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
| HF weights | `weights`, `model.yaml` (`model` path) |
| Tensor parallel | `model.yaml` (`tensor_parallel_size`) |
| KV cache fraction | `model.yaml` (`kv_cache_config`) |
| GPU | `resources.accelerator` |

For maximum throughput with a pre-built engine, use the `inflight_batcher_llm` C++ backend instead ([TensorRT-LLM backend quick start](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tensorrtllm_backend/README.html)).
