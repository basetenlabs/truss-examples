# baselabs/mammoth-1

Serve text/JSON Choice, Noul, and Score decisions on one H100.
Weights are pinned to [`8f9f86193e0ba44a3c5439fc59cc6851998bf8ae`](https://huggingface.co/baselabs/mammoth-1/tree/8f9f86193e0ba44a3c5439fc59cc6851998bf8ae) and loaded offline.
mammoth-1 is a full fine-tune of Qwen3.5-27B. It answers every question of a request from one shared-prefix prefill
without generating text. It runs on OneJev's `qev` runtime at
[`81ce62f`](https://github.com/OmniJev/OneJev/tree/81ce62f1597c91e46767d4d02d6ac2e18534fe94), with the checkpoint's
`serve.patch` applied at startup, the same way the checkpoint's `engine.py` runs it. The block-bidirectional
attention and slot readout are read from the checkpoint, so it is served the way it was trained.

The weights are public on Hugging Face under baselabs' license (see the model card); no access token is needed. This preset accepts text only. Image/video inputs are rejected. Requests whose state plus
longest question exceeds 32,768 tokens, or whose total exceeds 65,536 tokens, are rejected rather than truncated.
Inference is serialized; startup warms a decision before `/health` becomes ready.
The BF16 weights (about 55 GB) plus the fp32 LM head (about 5 GB) leave about 18 GB on the H100 for
batched question branches; branches that run out of memory fall back to OneJev's sequential path.
qev loads the model onto a single device, so more H100s would not add memory; a larger single GPU would.

```bash
truss push classifiers/mammoth-1/h100x1-bf16/default --remote <remote-name>
```

`POST /predict` accepts `state` and `questions`. `POST /v1/systemone` also requires
`model: mammoth-1` (or `baselabs/mammoth-1`). `/v1/models`, `/health`, and `/metrics` are
available under the deployment's `/sync` prefix.

CI runs all 231 public JevBench text cases. It does not measure throughput, concurrent
latency, or multimodal quality; the figures below come from a one-off replay of that suite.

One replica on one H100, replaying the same 231 public JevBench requests CI sends (one question each; median 179 input tokens, max 3,972) after 5 warm-up requests. Latency is end to end from a client outside Baseten, so it includes network round trips; inference is serialized, so throughput levels off once the GPU is saturated. Measured 2026-10-20.

| Concurrency | p50 | p90 | p99 | Requests/s | Correct |
|---|---|---|---|---|---|
| 1 | 0.42 s | 0.65 s | 0.80 s | 1.9 | 217/231 |
| 2 | 0.41 s | 0.76 s | 1.02 s | 4.1 | 217/231 |
| 4 | 0.69 s | 1.39 s | 1.98 s | 4.7 | 217/231 |
| 8 | 1.40 s | 2.54 s | 3.40 s | 4.7 | 217/231 |

CPU adapter tests use mocked inference:

```bash
PYTHONPATH=classifiers/mammoth-1/h100x1-bf16/default python -m pytest classifiers/mammoth-1/h100x1-bf16/default/tests
```