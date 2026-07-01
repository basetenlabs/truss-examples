# DiffusionGemma 26B A4B Instruct — FP8, H100:1

Serves `RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic` (FP8 weights, dynamic activations) on a single H100. DiffusionGemma is a discrete diffusion LLM: it denoises a 256-token canvas per block instead of decoding autoregressively, so responses stream in canvas-sized bursts and TTFT includes the first full canvas denoise.

vLLM support is not yet in a release; this preset builds vLLM from the DiffusionGemma PR branch ([vllm-project/vllm#45163](https://github.com/vllm-project/vllm/pull/45163), pinned at `d25326b1f`) by overlaying the branch onto the merge-base precompiled wheel. Once the PR ships in a release, swap to a stock `vllm/vllm-openai` image and remove the git/pip `build_commands`.

Benchmarked 2026-06-10 (aiperf, ISL 1024 / OSL 1024): 906 tok/s aggregate at concurrency 1 (960 tok/s per user including TTFT), 1584 tok/s at concurrency 8. `max-num-seqs 8` and `gpu-memory-utilization 0.85` are required headroom for diffusion warmup, which allocates `[seqs, canvas, vocab]` logits buffers.