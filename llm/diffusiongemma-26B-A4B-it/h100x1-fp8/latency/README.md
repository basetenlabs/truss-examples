# DiffusionGemma 26B A4B Instruct — FP8, H100:1

Serves `RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic` (FP8 weights, dynamic activations) on a single H100. DiffusionGemma is a discrete diffusion LLM: it denoises a 256-token canvas per block instead of decoding autoregressively, so responses stream in canvas-sized bursts and TTFT includes the first full canvas denoise.

DiffusionGemma support shipped in vLLM v0.28.0 ([vllm-project/vllm#45163](https://github.com/vllm-project/vllm/pull/45163), merged 2026-06-12); this preset uses the stock `vllm/vllm-openai:v0.28.0` image.

Benchmarked 2026-06-10 (aiperf, ISL 1024 / OSL 1024): 906 tok/s aggregate at concurrency 1 (960 tok/s per user including TTFT), 1584 tok/s at concurrency 8. `max-num-seqs 8` and `gpu-memory-utilization 0.85` are required headroom for diffusion warmup, which allocates `[seqs, canvas, vocab]` logits buffers.