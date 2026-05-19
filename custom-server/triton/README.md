# NVIDIA Triton Inference Server on Baseten

Examples for serving LLMs with [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server) on Baseten using [`docker_server`](https://docs.baseten.co/development/model/custom-server) (custom HTTP server), without a Truss `model.py` wrapper.

| Example | Backend | Baseten pattern |
|---------|---------|-----------------|
| [vllm-backend](./vllm-backend/) | [vLLM backend](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/vllm_backend/README.html) | HF weights via BDN → `model.json` |
| [tensorrtllm-backend](./tensorrtllm-backend/) | [TensorRT-LLM backend](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tensorrtllm_backend/README.html) | Pre-built engine + tokenizer via BDN → ensemble in `data/` |

## Shared conventions

- **`base_image`**: Official `nvcr.io/nvidia/tritonserver:*` images (`*-vllm-python-py3` or `*-trtllm-python-py3`).
- **`docker_server`**: `tritonserver` listens on port **8000** (not 8080 — reserved by Baseten's proxy).
- **`run_as_user_id: 1000`**: Required for NVIDIA images ([docs](https://docs.baseten.co/development/model/custom-server#non-root-user)).
- **`weights` (BDN)**: Model weights and engines mount at runtime; no download in `model.py`.
- **`data/`**: Triton model repository and startup scripts copied to `/app/data/`.
- **`predict_endpoint`**: Triton [generate extension](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/protocol/extension_generate.html) path; Baseten `/predict` forwards to it.
- **Health**: `/v2/health/ready` and `/v2/health/live`.
- **`call.py`**: Minimal HTTP client for `/environments/production/predict`.

## Quick compare

| | vLLM backend | TensorRT-LLM backend |
|---|--------------|----------------------|
| Container | `tritonserver:24.08-vllm-python-py3` | `tritonserver:24.07-trtllm-python-py3` |
| Weights | HF model weights | TRT-LLM engine + tokenizer |
| Triton model | Single `vllm_model` | `ensemble` (+ pre/post/tensorrt_llm) |
| Setup complexity | Lower (HF weights only) | Higher (engine build required) |
| Typical use | Fast path with vLLM inside Triton | Maximum throughput with TRT-LLM C++ runtime |

## Related examples in this repo

- [`templates/trt-llm`](../../templates/trt-llm/) — older Truss `model.py` that subprocesses Triton (not `docker_server`).
- [`custom-server/nemotron-parse-v1-2-vllm`](../nemotron-parse-v1-2-vllm/) — native vLLM OpenAI server (no Triton).
- [`vllm/`](../../vllm/) — Truss-native vLLM `AsyncLLMEngine`.

## References

- [Baseten custom Docker containers](https://docs.baseten.co/development/model/custom-server)
- [Truss configuration reference](https://docs.baseten.co/reference/truss-configuration)
- [Triton vLLM backend README](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/vllm_backend/README.html)
- [Triton TensorRT-LLM backend README](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tensorrtllm_backend/README.html)
