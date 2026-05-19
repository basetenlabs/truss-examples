# NVIDIA Triton Inference Server on Baseten

Examples for serving LLMs with [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server) on Baseten using [`docker_server`](https://docs.baseten.co/development/model/custom-server) and Triton's [**OpenAI-compatible frontend**](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client_guide/openai_readme.html).

The vLLM and TensorRT-LLM **backends** speak Triton's native APIs (for example the [generate extension](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/protocol/extension_generate.html)). For OpenAI-compatible routes (`/v1/chat/completions`, `/v1/completions`, `/v1/models`), these examples run NVIDIA's `openai_frontend/main.py`, which wraps Triton and the chosen backend.

| Example | Backend | OpenAI model name |
|---------|---------|-------------------|
| [vllm-backend](./vllm-backend/) | [vLLM backend](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/vllm_backend/README.html) | `qwen3-8b` |
| [tensorrtllm-backend](./tensorrtllm-backend/) | [TensorRT-LLM LLMAPI (PyTorch)](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tensorrtllm_backend/docs/llmapi.html) | `tensorrt_llm` |

## Shared conventions

- **`base_image`**: Official `nvcr.io/nvidia/tritonserver:*` images (`*-vllm-python-py3` or `*-trtllm-python-py3`).
- **`data/start-openai.sh`**: Starts `openai_frontend/main.py` on port **8000** (not 8080 — reserved by Baseten's proxy).
- **`run_as_user_id: 1000`**: Required for NVIDIA images ([docs](https://docs.baseten.co/development/model/custom-server#non-root-user)).
- **`weights` (BDN)**: Model weights and engines mount at runtime.
- **`data/model_repository/`**: Triton model configs; the **directory name** is the OpenAI `model` field in requests.
- **`predict_endpoint`**: `/v1/chat/completions` — Baseten `/predict` forwards here.
- **Health**: `/health/ready` on the OpenAI frontend.
- **`call.py`**: Uses the OpenAI Python SDK against `/environments/production/sync/v1`.

## Baseten routing

| Baseten endpoint | Maps to |
|------------------|---------|
| `/environments/production/predict` | `/v1/chat/completions` |
| `/environments/production/sync/v1/chat/completions` | `/v1/chat/completions` |
| `/environments/production/sync/v1/models` | `/v1/models` |

## Quick compare

| | vLLM backend | TensorRT-LLM backend |
|---|--------------|----------------------|
| Container | `tritonserver:25.08-vllm-python-py3` | `tritonserver:25.08-trtllm-python-py3` |
| Weights | `Qwen/Qwen3-8B` (HF) | `Qwen/Qwen3-8B` (HF, no TRT engine) |
| GPU | H100 | H100 |
| Backend runtime | vLLM in Triton | TensorRT-LLM PyTorch (`backend: pytorch`) |
| Setup | HF weights only | HF weights only (LLMAPI) |

## Related examples in this repo

- [`custom-server/nemotron-parse-v1-2-vllm`](../custom-server/nemotron-parse-v1-2-vllm/) — native vLLM OpenAI server (no Triton).
- [`templates/trt-llm`](../templates/trt-llm/) — legacy Truss `model.py` + raw Triton generate API.
- [`vllm/`](../vllm/) — Truss-native vLLM `AsyncLLMEngine`.

## References

- [Triton OpenAI-compatible frontend](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client_guide/openai_readme.html)
- [Baseten custom Docker containers](https://docs.baseten.co/development/model/custom-server)
- [Triton vLLM backend](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/vllm_backend/README.html)
- [Triton TensorRT-LLM backend](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tensorrtllm_backend/README.html)
