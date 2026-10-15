# D3-mini

Serve [vllm-sr/d3-mini](https://huggingface.co/vllm-sr/d3-mini) on one L4.
Supports Choice, Noul, and Score decisions over text/JSON, images, and video.

- Checkpoint and runtime: `0e78bbdbe3cf480ed1461c185c9e0c4fa9258011`.
  Startup verifies every manifest file; inference stays offline.
- Uses the D3 adapter's BF16 backbone, FP32 readout, serialized inference,
  and strict uploaded-media decoders. File paths and remote media URLs are rejected.
- `POST /predict` accepts `state` and `questions`; `POST /v1/systemone` also
  requires `model: d3-mini` (or `vllm-sr/d3-mini`). `/v1/models`, `/health`,
  and `/metrics` are available under the deployment's `/sync` prefix.

Deploy from the repository root with the intended workspace:

```bash
truss push classifiers/d3-mini/l4x1-bf16/default --remote <remote-name>
```

CI evaluates all 231 public JevBench text cases. This does not evaluate image/video
quality or throughput. Adapter/loader tests use mocked inference:

```bash
PYTHONPATH=classifiers/d3-mini/l4x1-bf16/default python -m pytest classifiers/d3-mini/l4x1-bf16/default/tests
```