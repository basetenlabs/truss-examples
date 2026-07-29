# Krea-2-Turbo performance package (for docs)

Prepared 2026-07-28 by FDE (Zak Keener) for the public-docs update on Krea-2-Turbo.

## Start here

1. `quality-review.html` — the main document. Grading key, performance summary table (what each configuration is, per-resolution latency, memory, quality verdict, what is Baseten-specific), cold-start table, and side-by-side image panels (click to zoom). Self-contained: the `images/` folder carries the panels.
2. `PERF-PROGRAM-SUMMARY.md` — the full report: claims the evidence supports (with the wording caveats built in), everything we tried and rejected with numbers, and the productization checklist.
3. `SCOREBOARD.md` — raw consolidated results table.
4. `estargz-pipeline.html` — cold-start mechanics explainer (how the image streaming works, with the before/after timeline).

## The two configs

Both deployable as-is with `truss push`:

- `configs/baseline-bf16.yaml` — the production lossless preset, identical to what is published in the model registry today (bf16, no acceleration, H100). Reference numbers: 1.542s @1024, 3.693s @1536, 7.308s @2048 server-side.
- `configs/optimized-fp8-cachedit.yaml` — the optimized preset (FP8 blockwise quantization + Cache-DiT step caching): 0.941s @1024, 2.218s @1536, 4.471s @2048, 27 GB peak memory. Quality: 6/9 prompts indistinguishable from baseline, 3/9 visibly different but equal quality, 0 unacceptable (the extended panel includes the text-rendering stress prompts Krea's research team suggested on 2026-07-19). Note in the config header: as measured it points at an experimental image tag built from an unmerged fork branch; the shipping tag will be cut after the fork PR merges. Do not document the experimental tag as the recommended image.

## Doc-claims guidance (evidence-backed, worded for external use)

- "Sub-second 1024px generation on a single H100" — supported (0.941s server-side, median of 5, 0.939-0.944s).
- "Faster than Krea's own API at its native 1.5K" — our 1536px is 2.218s server-side vs their advertised ~3s, but their number is end-to-end; if this claim is used, note the measurement-basis difference or hold it until an end-to-end comparison runs.
- "Only published native-2K number" — 4.471s @2048x2048 with named hardware; nobody else publishes a 2K latency with hardware attached.
- Quality wording: say "same quality as full-precision bf16 on our evaluation panel" — NOT "identical output" (not bit-exact).
- Cold start: "wakes from zero in about 2m40s in steady state" — with the caveat that the first wake after any deploy can take ~4-6 minutes (image optimization lag), documented in the pipeline explainer.
- Do not claim "faster than fal" — fal publishes no latency for this model.
- Not measured / not shipped: half-GPU (MIG) serving was tested and is rejected (3.7x slower, more expensive per image); torch.compile was tested and rejected (no speedup, +4 min cold start). Both results are documented with numbers if you want a "what we tried" sidebar.