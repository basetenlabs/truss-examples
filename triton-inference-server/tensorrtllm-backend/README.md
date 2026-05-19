# Triton + TensorRT-LLM — OpenAI-compatible API on Baseten

Deploy [NVIDIA Triton](https://github.com/triton-inference-server/server) with the [TensorRT-LLM backend](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tensorrtllm_backend/README.html) (`inflight_batcher_llm` **ensemble**) and Triton's [OpenAI-compatible frontend](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client_guide/openai_readme.html) on Baseten.

The TensorRT-LLM backend alone exposes Triton's generate API on `ensemble`. This example runs `openai_frontend/main.py` with `--backend tensorrtllm` for `/v1/chat/completions`.

## Prerequisites

1. Build a TensorRT-LLM engine for your GPU ([quick start](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tensorrtllm_backend/README.html)).
2. Upload engine artifacts via BDN (`weights`).
3. Update `config.yaml`: replace `hf://YOUR_ORG/your-trtllm-engine-repo@main` and align tokenizer weights with the engine.

Match the Triton image tag to your engine's TensorRT-LLM version ([support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)).

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

`TRTLLM_ORCHESTRATOR=1` is set for tensor-parallel engines (see [orchestrator mode](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tensorrtllm_backend/README.html#orchestrator-mode)).

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

On Baseten, use `/environments/production/predict` or `/environments/production/sync/v1/chat/completions`.

## Client

```bash
pip install openai
export BASETEN_API_KEY=...
export BASETEN_MODEL_ID=<model_id>
python call.py
```

Set `SERVED_MODEL` if you rename the ensemble directory in `data/model_repository/`.

## Performance tuning

Edit `data/model_repository/tensorrt_llm/config.pbtxt` (`max_num_sequences`, KV cache fractions, etc.). See [TRT-LLM model config](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tensorrtllm_backend/docs/model_config.html).
