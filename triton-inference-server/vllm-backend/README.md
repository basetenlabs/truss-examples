# Triton Inference Server — vLLM backend on Baseten

Deploy [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server) with the [vLLM backend](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/vllm_backend/README.html) as a [Baseten custom server](https://docs.baseten.co/development/model/custom-server).

Weights are delivered with [BDN](https://docs.baseten.co/reference/truss-configuration#weights) (`weights:`). The Triton model repository lives under `data/model_repository/`.

## Layout

```
vllm-backend/
├── config.yaml
├── call.py
└── data/
    └── model_repository/
        └── vllm_model/
            ├── config.pbtxt
            └── 1/model.json      # vLLM AsyncLLMEngine args
```

`model.json` points vLLM at `/models/llama`, where BDN mounts `meta-llama/Llama-3.2-1B-Instruct`.

## Deploy

```bash
cd triton-inference-server/vllm-backend
export HF_ACCESS_TOKEN=...   # used when truss prompts for hf_access_token secret
truss push
```

Use a GPU that fits the model (default: `A10G`). Align the Triton container tag with the vLLM version you need — see the [NGC tritonserver tags](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags) (`*-vllm-python-py3`).

## Inference

Triton exposes the [generate extension](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/protocol/extension_generate.html) on the vLLM model:

```bash
curl -X POST localhost:8000/v2/models/vllm_model/generate \
  -d '{"text_input": "Hello", "parameters": {"stream": false, "max_tokens": 32}}'
```

On Baseten, the same body is sent to `/environments/production/predict` (mapped to that route via `docker_server.predict_endpoint`).

Other Triton routes are available under `/environments/production/sync/`, for example:

- `/environments/production/sync/v2/health/ready`
- `/environments/production/sync/v2/models/vllm_model/generate`

## Client

```bash
pip install httpx
export BASETEN_API_KEY=...
export BASETEN_MODEL_ID=<model_id>
python call.py
```

## Customize

| Goal | Where to change |
|------|-----------------|
| Different HF model | `weights`, `data/.../1/model.json` (`model` path) |
| vLLM engine args | `data/model_repository/vllm_model/1/model.json` |
| Triton / vLLM version | `base_image.image` |
| GPU | `resources.accelerator` |
