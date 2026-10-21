# baselabs/elephant-1

Serve text/JSON Choice, Noul, and Score decisions on one H100.
Weights are pinned to [`14537f08b0374edf9867ed0a4d2bd8df59e0400c`](https://huggingface.co/baselabs/elephant-1/tree/14537f08b0374edf9867ed0a4d2bd8df59e0400c) and loaded offline.
elephant-1 is a full fine-tune of Qwen3.5-9B. It answers every question of a request from one shared-prefix prefill
without generating text. It runs on OneJev's `qev` runtime at
[`81ce62f`](https://github.com/OmniJev/OneJev/tree/81ce62f1597c91e46767d4d02d6ac2e18534fe94), with the checkpoint's
`serve.patch` applied at startup, the same way the checkpoint's `engine.py` runs it. The block-bidirectional
attention and slot readout are read from the checkpoint, so it is served the way it was trained.

The weights are public on Hugging Face under baselabs' license (see the model card); no access token is needed. This preset accepts text only. Image/video inputs are rejected. Requests whose state plus
longest question exceeds 32,768 tokens, or whose total exceeds 65,536 tokens, are rejected rather than truncated.
Inference is serialized; startup warms a decision before `/health` becomes ready.

```bash
truss push classifiers/elephant-1/h100x1-bf16/default --remote <remote-name>
```

`POST /predict` accepts `state` and `questions`. `POST /v1/systemone` also requires
`model: elephant-1` (or `baselabs/elephant-1`). `/v1/models`, `/health`, and `/metrics` are
available under the deployment's `/sync` prefix.

CI runs all 231 public JevBench text cases. It does not measure throughput, concurrent
latency, or multimodal quality; the figures below come from a one-off replay of that suite.

One replica on one H100, replaying the same 231 public JevBench requests CI sends (one question each; median 179 input tokens, max 3,972) after 5 warm-up requests. Latency is end to end from a client outside Baseten, so it includes network round trips; inference is serialized, so throughput levels off once the GPU is saturated. Measured 2026-10-20.

| Concurrency | p50 | p90 | p99 | Requests/s | Correct |
|---|---|---|---|---|---|
| 1 | 0.34 s | 0.42 s | 0.50 s | 2.6 | 207/231 |
| 2 | 0.33 s | 0.40 s | 0.54 s | 5.8 | 207/231 |
| 4 | 0.37 s | 0.58 s | 0.74 s | 9.6 | 207/231 |
| 8 | 0.72 s | 1.06 s | 1.31 s | 9.8 | 207/231 |

CPU adapter tests use mocked inference:

```bash
PYTHONPATH=classifiers/elephant-1/h100x1-bf16/default python -m pytest classifiers/elephant-1/h100x1-bf16/default/tests
```