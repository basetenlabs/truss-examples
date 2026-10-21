# baselabs/mammoth-1

Serve text/JSON Choice, Noul, and Score decisions on one H100.
Weights are pinned to [`8f9f86193e0ba44a3c5439fc59cc6851998bf8ae`](https://huggingface.co/baselabs/mammoth-1/tree/8f9f86193e0ba44a3c5439fc59cc6851998bf8ae).
mammoth-1 is a full fine-tune of Qwen3.5-27B that answers every question of a request with a probability per option,
without generating text. It is served in BF16 by Infinity v3
(`ghcr.io/michaelfeil/infinity_v3:nvidia-3.0.0-alpha.9`) with its OneJev decision protocol; the image runs unmodified.

The weights are public on Hugging Face under baselabs' license (see the model card); no access token is needed.

```bash
truss push classifiers/mammoth-1/h100x1-bf16/default --remote <remote-name>
```

`POST /v1/systemone` accepts `state`, `questions`, and `model: mammoth-1`. The endpoint is under the
deployment's `/sync` prefix; `/health` reports readiness.

CI runs all 231 public JevBench text cases. It does not measure throughput, concurrent latency, or multimodal quality.

Latency and throughput (Infinity v3):

One replica on one H100 (`predict_concurrency: 32`), replaying the 231 public JevBench requests CI sends (one question each; median 179 input tokens, max 3,972) with JevBench's own adapter and scorer, after 5 warm-up requests. Latency is end to end from a client outside Baseten, so it includes network round trips. Measured 2026-10-20.

| Concurrency | p50 | p90 | p99 | Requests/s | Correct |
|---|---|---|---|---|---|
| 1 | 0.35 s | 0.64 s | 0.87 s | 2.4 | 217/231 |
| 2 | 0.34 s | 0.75 s | 1.06 s | 4.6 | 217/231 |
| 4 | 0.36 s | 1.34 s | 2.07 s | 6.7 | 217/231 |
| 8 | 0.59 s | 2.21 s | 3.77 s | 7.6 | 217/231 |
| 16 | 0.81 s | 3.59 s | 4.63 s | 9.1 | 217/231 |
| 32 | 1.18 s | 6.28 s | 8.21 s | 10.0 | 217/231 |