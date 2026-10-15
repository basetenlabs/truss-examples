# Cloudflare/clef-flash

Serve text/JSON Choice, Noul, and Score decisions on one H100.
Weights and inference code are pinned to [`8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3`](https://huggingface.co/Cloudflare/clef-flash/tree/8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3) and loaded offline.
Uses the upstream joint schema head and BF16 loader without changing answer scoring.

This preset accepts text only. Image/video inputs are rejected. Requests exceeding
16,384 encoded request tokens are rejected rather than silently truncated.
Inference is serialized; startup warms a decision before `/health` becomes ready.

```bash
truss push classifiers/clef-flash/h100x1-bf16/default --remote <remote-name>
```

`POST /predict` accepts `state` and `questions`. `POST /v1/systemone` also requires
`model: clef-flash` (or `Cloudflare/clef-flash`). `/v1/models`, `/health`, and `/metrics` are
available under the deployment's `/sync` prefix.

CI runs all 231 public JevBench text cases. It does not measure throughput or
multimodal quality. CPU adapter tests use mocked inference:

```bash
PYTHONPATH=classifiers/clef-flash/h100x1-bf16/default python -m pytest classifiers/clef-flash/h100x1-bf16/default/tests
```