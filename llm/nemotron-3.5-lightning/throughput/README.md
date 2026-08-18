# NVIDIA Nemotron 3.5 Lightning 30B-A3B (NVFP4) — Throughput Template

NVIDIA Nemotron 3.5 Lightning is a hybrid **Mamba-2 + MoE + Attention**
reasoning model with Multi-Token Prediction, **30B total / 3B active**
parameters, and a native **1,048,576-token context window**. This Truss serves
the NVFP4 checkpoint with **vLLM v0.27.0** on one H100.

The throughput preset uses DFlash speculative decoding (K=3) with NVIDIA's
W4A16_NVFP4 drafter, Humming MoE and quantized-linear kernels, FlashInfer
Mamba, FP16 Mamba state with stochastic rounding, aligned prefix caching,
`max-num-seqs=512`, and `max-num-batched-tokens=32768`.

## Public checkpoints

Both checkpoints are official, public NVIDIA Hugging Face repositories mounted
through Baseten Delivery Network (BDN) and pinned to immutable commits:

- Verifier:
  `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4@0dcd680e5585c791728c83342b311d0a0026dbeb`
- DFlash drafter:
  `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash@7fc1f1ff4b82b917efbd0710df0872c2bb89caa5`

The public verifier contains NVIDIA's current shard 52 object, rather than the
divergent and damaged shard previously found in the private Baseten mirror.
The old baked private image is no longer used.

The registry template declares an `hf_access_token` secret in keeping with
registry convention. A Hugging Face read token is sufficient; neither repo is
gated.

## Endpoint

The deployment exposes OpenAI-compatible chat completions at
`/v1/chat/completions`, with served model name
`nvidia/nemotron-3.5-lightning-nvfp4`. Tool-bearing requests are enabled with
vLLM's `nemotron_v3` reasoning parser and `qwen3_coder` tool-call parser.

## Verification

The exact public-HF 1M configuration was deployed in the FDE Internal
workspace as model `qel8r723`, production deployment `3yv4n84` on 2026-08-11:

- Deployment became ACTIVE on one H100, and `/v1/models` reported
  `max_model_len: 1048576`.
- vLLM allocated 43.31 GiB of KV cache: 7,404,361 tokens, or 7.06 concurrent
  full-window requests.
- The shared-47K-prefix / unique-2K-suffix / forced-1K-output workload at
  concurrency 512 measured **6,165.1 client-observed output tokens/s**. This
  was 1.21% below the otherwise matching 52K public-HF configuration's
  6,240.9 tokens/s.
- The 1M run met the 6,000-token/s throughput threshold but did not pass the
  strict zero-failure gate: one of 1,101 requests failed with a connection
  reset. Runtime logs also showed two CUDA allocator OOM warnings and one
  preemption while KV-cache use peaked at 99.9%.
- Prefix-cache hits were typically 92-94%, and DFlash mean acceptance length
  reached 3.84 tokens. The replica remained ACTIVE and drained the workload.

The concurrency-512 benchmark is an edge-of-memory stress setting, not a
production-safe concurrency recommendation. It measures the overhead of
enabling the 1M ceiling on roughly 50K-token sequences; requests that actually
consume the full context window require a separate long-prefill benchmark.