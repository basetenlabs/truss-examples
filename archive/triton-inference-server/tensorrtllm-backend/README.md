# Triton + TensorRT-LLM (PyTorch / LLMAPI) — OpenAI API on Baseten

Deploy [NVIDIA Triton](https://github.com/triton-inference-server/server) with the TensorRT-LLM [**LLMAPI PyTorch backend**](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tensorrtllm_backend/docs/llmapi.html) and Triton's [OpenAI-compatible frontend](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client_guide/openai_readme.html) on Baseten.

This path uses `backend: pytorch` in `model.yaml` and loads [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) from BDN at runtime. **No pre-built TensorRT engine is required** (unlike the `inflight_batcher_llm` C++ backend).

## NVIDIA backend files (provenance)

The files under `data/model_repository/tensorrt_llm/` (`config.pbtxt`, `1/model.py`, `1/helpers.py`) are copied from NVIDIA’s **LLMAPI Python backend** reference for Triton 25.08 (same lineage as `base_image` `nvcr.io/nvidia/tritonserver:25.08-trtllm-python-py3`). See the [LLMAPI workflow](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tensorrtllm_backend/docs/llmapi.html) and the [`tensorrtllm_backend`](https://github.com/triton-inference-server/tensorrtllm_backend) repo for upstream sources.

When you bump the `base_image` tag, diff your copies against the new release’s LLMAPI reference and re-copy if NVIDIA changed inputs, outputs, or the Python backend contract. Local edits here can drift from upstream.

## Triton `instance_group` uses `KIND_CPU`

`config.pbtxt` schedules the Python backend on CPU (`kind: KIND_CPU`). That is normal for the LLMAPI path: Triton’s Python process orchestrates requests while TensorRT-LLM (`backend: pytorch` in `model.yaml`) runs the model on GPU. To use multiple GPUs, raise `tensor_parallel_size` in `model.yaml` and configure `gpu_device_ids` in `config.pbtxt` per [NVIDIA’s multi-instance docs](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tensorrtllm_backend/docs/llmapi.html).

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

OpenAI chat completions — use Triton model name `tensorrt_llm`. Use `max_tokens` (the OpenAI SDK and this repo’s `call.py` use that name). Some clients send `max_completion_tokens`; the OpenAI frontend maps it to Triton’s `max_tokens` input.

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "tensorrt_llm",
    "messages": [{"role": "user", "content": "What is ML?"}],
    "max_tokens": 64
  }'
```

Streaming requires Triton **decoupled** mode (`triton_config.decoupled: true` in `model.yaml`, already set in this example). Without it, `model.py` rejects `stream: true`.

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "tensorrt_llm",
    "messages": [{"role": "user", "content": "What is ML?"}],
    "max_tokens": 64,
    "stream": true
  }'
```

The LLMAPI `config.pbtxt` also accepts OpenAI-frontend aliases (`temperature`, `max_tokens`, `stream`, etc.) via `helpers.py`.

## Client

```bash
pip install openai
export BASETEN_API_KEY=...
export BASETEN_MODEL_ID=<model_id>
python call.py

# SSE streaming (requires decoupled: true in model.yaml)
STREAM=1 python call.py
```

## Customize

| Goal | Where to change |
|------|-----------------|
| HF weights | `weights`, `model.yaml` (`model` path) |
| Tensor parallel | `model.yaml` (`tensor_parallel_size`) |
| KV cache fraction | `model.yaml` (`kv_cache_config`) |
| GPU | `resources.accelerator` |

For maximum throughput with a pre-built engine, use the `inflight_batcher_llm` C++ backend instead ([TensorRT-LLM backend quick start](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tensorrtllm_backend/README.html)).
