# Krea-2-Turbo Perf Program Scoreboard

Goal: a defensible "runs better on Baseten" claim. Baseline = prod lossless preset
(bf16, H100, no cache-dit, no compile), model 31d1dyj3 / deployment q0p97v1.

Competitive anchors (2026-07-16): SGLang reference 1.56s @1024 on H200; Krea's own
API turbo ~3s at 1.5K native ($0.015/img); fal serves open weights at $0.008/MP with
no published latency. Nobody publishes 2K numbers with named hardware except us.

## Latency (median server-side inference_time_s, fixed prompt, seed 42)

| Branch | Deployment | 1024x1024 | 1536x1536 | 2048x2048 | Quality vs baseline | Status |
|---|---|---|---|---|---|---|
| baseline (prod) | 31d1dyj3/q0p97v1 | 1.542 | 3.693 | 7.308 | (reference) | measured 2026-07-17 04:04 UTC |
| cache-dit (defaults rdt=0.24) | qjd6m7gq/qrpx8p9 | 1.225 (1.26x) | 2.851 (1.30x) | 5.635 (1.30x) | 6/9 negligible, 3/9 visible, 0 unacceptable (9-prompt panel incl Krea-requested text stress) | COMPLETE. SHIP recommendation: default fast path |
| cache-dit aggressive (rdt=0.35 via SGLANG_CACHE_DIT_RDT) | qjd6m7gq/qz4d8p8 | ~1.43-1.50x vs baseline | | | 2 negligible / 3 visible / 1 UNACCEPTABLE | opt-in max-speed mode at most; do NOT default. NOTE: SGLANG_CACHE_DIT_RESIDUAL_DIFF_THRESHOLD is a silent no-op in v1.3; real knob is SGLANG_CACHE_DIT_RDT |
| torch-compile | 31d7kr93/31l9mlg | 1.544 (noise) | 3.642 (noise) | 7.25 (noise) | near-lossless (fp drift only) | COMPLETE, NEGATIVE: no steady-state gain (inductor re-picks cuBLAS; attention already FA3), +~4min compile per cold start, 47s first-request stall. Do not productize. Composition probe (compile+cache-dit): DEPLOY_FAILED, early init crash exit 128 on v1.3 — not a drop-in combo (academic; combined config excludes compile) |
| cold-start (latency unchanged) | q414k6kq/wozdjzk | n/a | n/a | n/a | n/a | weights 62->36GB shipped; streaming blocked on runtime team |
| fp8 blockwise | w7p2k7dw/324eo51 | 1.194 (1.29x) | 2.856 (1.29x) | 5.821 (1.26x) | 6/9 negligible, 3/9 visible, 0 unacceptable; tiny-text softening worst under fp8 (q8 URL) | COMPLETE. Peak mem 27.1GB @1024 (-28%). Ship as lossy preset candidate (full H100). 196 layers quantized, to_gate/text-fusion/embedders bf16. MIG follow-up: REJECTED (see combined-mig row) |
| **combined fp8 + cache-dit** | q95g6o9w/wdld67n | **0.941 (1.64x)** | **2.218 (1.66x)** | **4.471 (1.63x)** | 6/9 negligible, 3/9 visible, 0 unacceptable on the extended panel; primary text/layout preserved everywhere, one smallest-text URL softens | COMPLETE. THE SHIPPABLE PRESET. Sub-second @1024 (5/5 runs 0.939-0.944). Mem ~= fp8 (27.2GB @1024). Both engines log-verified. Add ignore_patterns to prod config |
| combined on H100 MIG 40GB | qelmmrr3 (krea2-exp-combined-mig) | 3.478 (3.7x SLOWER than full-H100 combined) | ~8s intermittent 500s (libtorch allocator NVML bug, not OOM) | 500 always | equal tier, same-seed composition differs | REJECTED: 0.58x price x 3.7x latency = ~2.1x cost/image; 1024-only; warmup must be 1024-only; failures clean (no crash). Bonus: fork PR #65 fixes MIG boot for ALL diffusion models |

## Cold start (wake-from-zero to first 200)

| Branch | Image pull | Weights (BDN) | Load+JIT+warmup | Total pod-level | Notes |
|---|---|---|---|---|---|
| baseline (measured 2026-07-07 wake) | 4m32s (19GB, ~70MB/s, estargz tag but lazy pull NOT engaged) | 23s (62GB @2.7GB/s) | ~80s | 6m20s | image pull = 72% |
| cold-start branch (wake0, 2026-07-17) | 4m17s (un-optimized `-estargz` tag, full pull) | ~2s (36GB node-cached; cold-pull est ~13s after turbo.safetensors excluded) | ~98s (import 26s + init 13s + load 49s + warmup 9.5s) | ~6m0s | measured during the ~25min post-deploy optimization window |
| cold-start FINAL (steady state, measured) | **5.1s** (`-estargz-optimized`, 17.08GB streamed; optimizer runs ASYNC ~20-25min after deploy, so the FIRST wake per deploy still full-pulls ~4min) | ~2s cached / ~13s cold | import 78s (on-demand .so faulting; landmark coverage gap, ~30-50s recoverable) + load 45s (text_encoder slow path 22-27s) + warmup | **~2m40s** | remaining asks: (1) runtime team: pre-warm/shorten optimization lag, (2) landmark coverage for the base image, (3) fork: text_encoder loader + /health_generate readiness fix |

## Rules
- All numbers from `harness/bench.py` (same prompts/seeds/counts) unless noted.
- Lossy branches (cache-dit, fp8) must include the 6-image quality panel vs baseline
  before any number goes in a customer-facing claim.
- Experiment deployments: internal `baseten` org, deployment name `krea2-exp-<branch>`,
  torn down when branch concludes.
