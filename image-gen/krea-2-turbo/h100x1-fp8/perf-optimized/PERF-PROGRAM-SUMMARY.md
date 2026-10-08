# Krea-2-Turbo on Baseten: Performance Program Summary

Date: 2026-07-17. Owner: Zak Keener (FDE). Scope: four parallel optimization branches plus a combined validation against the production Krea-2-Turbo deployment (bf16, 1x H100 80GB, SGLang diffusion stack, image v1.3).

All latency figures are median server-side inference_time_s from a fixed-prompt, fixed-seed harness (customers/krea/harness/bench.py), measured on dedicated single-H100 deployments in the internal org. Quality was evaluated on a six-prompt fixed-seed panel against the bf16 baseline; images are preserved for review (quality-review.html).

## Headline result

Krea-2-Turbo generates a 1024x1024 image in 0.941 seconds server-side on a single H100 (5/5 runs within 0.939-0.944s), using FP8 blockwise quantization plus Cache-DiT step caching. Quality on the evaluation panel (extended to 9 prompts on 2026-07-19 with text-rendering stress tests requested by Krea research): 6 of 9 indistinguishable from bf16, 3 of 9 visibly different but equal quality, 0 unacceptable. All hierarchical text, dates, prices, and layout constraints render correctly under every shipping preset; the one preset-attributable text effect is softening of the smallest on-image text (a footer URL), worst under FP8.

## Results table

| Configuration | 1024x1024 | 1536x1536 | 2048x2048 | Peak mem @1024 | Quality vs bf16 |
|---|---|---|---|---|---|
| Baseline bf16 (prod today) | 1.542s | 3.693s | 7.308s | 37.7 GB | reference |
| Cache-DiT (rdt 0.24) | 1.225s | 2.851s | 5.635s | ~37.7 GB | 5/6 negligible, 1/6 visible |
| FP8 blockwise | 1.194s | 2.856s | 5.821s | 27.1 GB | 5/6 negligible, 1/6 composition divergence |
| Combined FP8 + Cache-DiT | 0.941s | 2.218s | 4.471s | 27.2 GB | 5/6 negligible, 1/6 composition divergence |
| torch.compile (rejected) | 1.544s | 3.642s | 7.25s | n/a | near-lossless, zero gain |

Cold start (scale from zero to serving): ~2m40s steady state, of which image pull is 5.1s via Baseten image streaming. Caveat: the streamable image variant is produced asynchronously ~20-25 minutes after each deploy; the first wake inside that window pays a ~4-minute full pull.

## Claims the evidence supports

1. Sub-second Krea-2-Turbo at 1024px on a single H100 (0.941s server-side, named hardware, reproducible). No other provider publishes any Krea-2 latency with hardware attached.
2. Faster than Krea's own API at its native resolution: 2.218s at 1536px vs the Krea API's advertised ~3s at 1.5K for its hosted Turbo. (Their figure is end-to-end-ish; ours is server-side. An end-to-end comparison should be run before this goes in print.)
3. The only published native-2K number in the market: 4.471s at 2048x2048.
4. Full-precision option matches the SGLang reference: our bf16 baseline (1.542s) matches the official cookbook figure measured on an H200 (1.56s), on cheaper hardware.
5. Unit economics: at 0.941s/image, one H100 sustains ~3,800 images/hour, roughly $0.001-0.002/image at high utilization vs fal's $0.008/MP and Krea/Replicate's $0.015/image list prices.
6. Cost shape, tested and rejected: FP8's memory footprint fits a half-GPU H100 MIG 40GB slice, but measured latency there is 3.478s at 1024 (3.7x slower than full H100), making MIG ~2.1x more expensive per image despite the 0.58x slice price; 1536+ is unreliable on MIG due to a libtorch allocator bug. Do not offer a MIG preset. Incidental win: fork PR #65 (one-line fix) makes sglang-diffusion bootable on MIG at all.
7. Scale-to-zero wake in ~2m40s.

Claim hygiene: quality wording should be "same quality" (panel-verified) not "identical output"; note the smallest-text edge case (tiny URLs soften under acceleration; primary text unaffected) (runs are not bit-exact vs bf16); do not claim "faster than fal" (they publish no number) without running their endpoint through an InferBench-style methodology; Krea 2 Community License obligations (content filtering, enterprise licensing thresholds) apply to how customer-facing claims are packaged.

## What did not work (measured, saved for posterity)

- torch.compile (max-autotune): zero steady-state gain on this model (inductor selects the same cuBLAS kernels; attention is already on tuned FA3), while adding ~4 minutes of compile to every cold start and a 47s first-request stall. Rejected.
- Cache-DiT rdt=0.35 (aggressive): 1.43-1.50x but 1 of 6 panel images unacceptable. Quality cliff sits between rdt 0.24 and 0.35. Opt-in at most.
- JIT cache persistence via b10cache: persistable artifacts on the bf16 path total ~400 KB; the real warmup cost is non-cacheable GPU state. Not pursued.
- compile + cache-dit: crashes at init on v1.3. Recorded as a known-bad combination.

## Productization checklist

1. Zak: independent quality review of the image panels (quality-review.html), which gates everything below.
2. sglang fork: take PR #64 (FP8 Krea-2 preset, currently draft) through review; cut the next production image tag from b10-main after merge.
3. model-registry: add the combined preset as a new entry/preset (for example preset:turbo) alongside the untouched lossless preset; both entries should gain weights ignore_patterns for the redundant 26 GB turbo.safetensors (validated: loader never reads it).
4. Fork fix candidates, ordered by value: /health_generate readiness stub (replicas currently report Ready before load/warmup); text_encoder load path (~22-27s, slow native loader); default-seed behavior (unseeded requests appear pinned to seed 42, so identical prompts return identical images).
5. Infra follow-ups (owner Zak, filed separately): the cru-us-east1 container-init incident (INC-5467, evidence in configs/combined/infra-evidence.md); the image-optimizer lag pre-warm request (draft in configs/cold-start/infra-ticket-draft.md).
6. Teardown after quality sign-off: experiment models krea2-exp-{cachedit, compile, coldstart, fp8, combined} (all scaled to zero; the combined deployment stays reproducible from configs/combined/config.yaml).

## Artifacts

- SCOREBOARD.md: consolidated numbers.
- configs/<branch>/RESULTS.md: per-branch full detail and evidence.
- quality-review.html: side-by-side image panels for human review.
- harness/bench.py: the shared benchmark (fixed prompts and seeds; reproducible).
