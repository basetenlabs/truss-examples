# Mapika/decider-4b

Serve text/JSON Choice, Noul, and Score decisions on one L4.
Weights and inference code are pinned to [`99d7fd5bfcc4d11ba37239dafe0ed653803819d3`](https://huggingface.co/Mapika/decider-4b/tree/99d7fd5bfcc4d11ba37239dafe0ed653803819d3) and loaded offline.
Uses upstream per-type temperatures from `decider_config.json`; eager inference preserves the released prompt layout.

This preset accepts text only. Image/video inputs are rejected. Requests exceeding
32,768 context tokens (including the `Context:\n` prefix) are rejected rather than silently truncated.
Inference is serialized; startup warms a decision before `/health` becomes ready.

```bash
truss push classifiers/decider-4b/l4x1-bf16/default --remote <remote-name>
```

`POST /predict` accepts `state` and `questions`. `POST /v1/systemone` also requires
`model: decider-4b` (or `Mapika/decider-4b`). `/v1/models`, `/health`, and `/metrics` are
available under the deployment's `/sync` prefix.

CI runs all 231 public JevBench text cases. It does not measure throughput or
multimodal quality. CPU adapter tests use mocked inference:

```bash
PYTHONPATH=classifiers/decider-4b/l4x1-bf16/default python -m pytest classifiers/decider-4b/l4x1-bf16/default/tests
```