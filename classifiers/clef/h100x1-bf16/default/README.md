# Cloudflare/clef

Serve text/JSON Choice, Noul, and Score decisions on one H100.
Weights and inference code are pinned to [`0b331204bb13fbd2ca93a64df1956f5b55478ce5`](https://huggingface.co/Cloudflare/clef/tree/0b331204bb13fbd2ca93a64df1956f5b55478ce5) and loaded offline.
Uses the upstream joint schema head and BF16 loader without changing answer scoring.

This preset accepts text only. Image/video inputs are rejected. Requests exceeding
16,384 encoded request tokens are rejected rather than silently truncated.
Inference is serialized; startup warms a decision before `/health` becomes ready.

```bash
truss push classifiers/clef/h100x1-bf16/default --remote <remote-name>
```

`POST /predict` accepts `state` and `questions`. `POST /v1/systemone` also requires
`model: clef` (or `Cloudflare/clef`). `/v1/models`, `/health`, and `/metrics` are
available under the deployment's `/sync` prefix.

CI runs all 231 public JevBench text cases. It does not measure throughput or
multimodal quality. CPU adapter tests use mocked inference:

```bash
PYTHONPATH=classifiers/clef/h100x1-bf16/default python -m pytest classifiers/clef/h100x1-bf16/default/tests
```