# Triton Inference Server — TensorRT-LLM backend on Baseten

Deploy [NVIDIA Triton](https://github.com/triton-inference-server/server) with the [TensorRT-LLM backend](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tensorrtllm_backend/README.html) (`inflight_batcher_llm` ensemble) as a [Baseten custom server](https://docs.baseten.co/development/model/custom-server).

This example uses:

- **BDN** (`weights:`) for the TRT-LLM engine and Hugging Face tokenizer
- **`data/model_repository/`** for the ensemble, preprocessing, postprocessing, and `tensorrt_llm` configs
- **`data/start-triton.sh`** to copy the engine into the repo and set `triton_tokenizer_repository` before starting `tritonserver`

## Prerequisites

1. Build a TensorRT-LLM engine for your GPU topology (see [TensorRT-LLM docs](https://github.com/NVIDIA/TensorRT-LLM) and the Triton [quick start](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tensorrtllm_backend/README.html)).
2. Upload the engine artifacts to Hugging Face or S3.
3. Update `config.yaml`:
   - Replace `hf://YOUR_ORG/your-trtllm-engine-repo@main` with your engine source.
   - Point `TOKENIZER_DIR` / tokenizer `weights` at a repo that matches the engine.

The Triton image tag (`24.07-trtllm-python-py3`) must match the TensorRT-LLM version used to build the engine ([support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)).

## Layout

```
tensorrtllm-backend/
├── config.yaml
├── call.py
└── data/
    ├── start-triton.sh
    └── model_repository/
        ├── ensemble/
        ├── preprocessing/1/model.py
        ├── postprocessing/1/model.py
        └── tensorrt_llm/          # engine copied here at startup
```

## Deploy

```bash
cd triton-inference-server/tensorrtllm-backend
# Edit config.yaml: engine weights source + tokenizer repo
truss push
```

`run_as_user_id: 1000` is set because NVIDIA Triton images run as UID 1000 ([Baseten custom server docs](https://docs.baseten.co/development/model/custom-server#non-root-user)).

## Inference

The ensemble model uses Triton's generate API:

```bash
curl -X POST localhost:8000/v2/models/ensemble/generate \
  -d '{"text_input": "What is ML?", "max_tokens": 32, "bad_words": "", "stop_words": ""}'
```

On Baseten, POST the same JSON to `/environments/production/predict`.

For multi-GPU tensor parallel engines, use `mpirun` / `launch_triton_server.py` as in NVIDIA's docs and adjust `docker_server.start_command` accordingly.

## Client

```bash
pip install httpx
export BASETEN_API_KEY=...
export BASETEN_MODEL_ID=<model_id>
python call.py
```

## Performance tuning

Edit `data/model_repository/tensorrt_llm/config.pbtxt` parameters such as `max_num_sequences`, `kv_cache_free_gpu_mem_fraction`, and `max_tokens_in_paged_kv_cache`. See [TRT-LLM model config](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tensorrtllm_backend/docs/model_config.html) and the legacy [`templates/trt-llm`](../../templates/trt-llm/TRT-LLM-README.md) notes in this repo.
