# baselabs/elephant-1

Serve text/JSON Choice, Noul, and Score decisions on one H100.
Weights are pinned to [`14537f08b0374edf9867ed0a4d2bd8df59e0400c`](https://huggingface.co/baselabs/elephant-1/tree/14537f08b0374edf9867ed0a4d2bd8df59e0400c).
elephant-1 is a full fine-tune of Qwen3.5-9B that answers every question of a request with a probability per option,
without generating text. It is served in BF16 by Infinity v3
(`ghcr.io/michaelfeil/infinity_v3:nvidia-3.0.0-alpha.9`) with its OneJev decision protocol; the image runs unmodified.

The weights are public on Hugging Face under baselabs' license (see the model card); no access token is needed.

```bash
truss push classifiers/elephant-1/h100x1-bf16/default --remote <remote-name>
```

`POST /v1/systemone` accepts `state`, `questions`, and `model: elephant-1`. The endpoint is under the
deployment's `/sync` prefix; `/health` reports readiness.

CI runs all 231 public JevBench text cases. It does not measure throughput, concurrent latency, or multimodal quality.

Latency and throughput (Infinity v3):

One replica on one H100 (`predict_concurrency: 32`), replaying the 231 public JevBench requests CI sends (one question each; median 179 input tokens, max 3,972) with JevBench's own adapter and scorer, after 5 warm-up requests. Latency is end to end from a client outside Baseten, so it includes network round trips. Measured 2026-10-20.

| Concurrency | p50 | p90 | p99 | Requests/s | Correct |
|---|---|---|---|---|---|
| 1 | 0.30 s | 0.40 s | 0.54 s | 3.1 | 208/231 |
| 2 | 0.30 s | 0.40 s | 0.54 s | 6.1 | 208/231 |
| 4 | 0.30 s | 0.50 s | 0.72 s | 11.6 | 208/231 |
| 8 | 0.33 s | 0.82 s | 1.20 s | 17.8 | 208/231 |
| 16 | 0.43 s | 1.28 s | 1.51 s | 23.2 | 208/231 |
| 32 | 0.62 s | 2.05 s | 2.63 s | 28.3 | 208/231 |