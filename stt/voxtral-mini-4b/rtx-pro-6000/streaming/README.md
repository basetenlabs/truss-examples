# Voxtral Mini 4B Realtime, RTX PRO 6000 streaming

## Startup and JIT caches

`VLLM_CACHE_ROOT`, `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` stay unset, so the platform's
startup cache puts vLLM's torch.compile output and the Triton and Inductor caches under `/bdn/`.
It saves them once the first replica is ready and restores them into later pods. The preset used
to point all three at the shared `/cache/org` mount. That bypasses the startup cache, and the
same setting hung startup on another preset until the 3000 s check killed it (#1077, #1079).

Measured in the Model Scout startup study (2026-11-13,
[PR #1272](https://github.com/basetenlabs/model-registry/pull/1272), Actions run 43519478188). Both arms
ran a cold first boot on ali-ue1-prod-1 with an empty startup cache and no prior `/cache/org` entry.
Source: pod logs in ClickHouse `baseten.model_logs`.

| Phase | `/cache/org` (old) | startup cache (now) |
| --- | ---: | ---: |
| compute acquired -> app ready | 181.2 s | 167.8 s |
| vLLM args -> app ready | 79.9 s | 65.3 s |
| Inductor compile (range 1-8192) | 23.6 s | 10.2 s |
| torch.compile total | 33.8 s | 20.0 s |
| weights pull / load | 40.7 / 4.9 s | 42.1 / 4.6 s |
| CUDA-graph capture | 1 s | 1 s |
| startup-cache publish after ready | 216 B (+ 102 KiB driver) | 62.4 MiB (+ 102 KiB driver) |

The first boot is 13.4 s faster because writing the compile to the network mount is slower than
local disk. That cause is inferred, not measured. The larger gain is on later pods, which restore
the 62 MiB publish instead of compiling again. That restore was not measured: this preset has no
restart diagnostic, and batch-bench runs one replica. To revert, set the three variables back to
`/cache/org/vllm`, `/cache/org/inductor` and `/cache/org/triton`.

Serving and quality have no same-run baseline: the `/cache/org` arm's benchmark job ran past its
deadline and was cancelled with no results. Against main's stored run (`run-6a6f62163ffb`,
2026-10-17), the pipecat streaming cell is unchanged: client p50/p90 791/1,039 ms vs 793/1,023 ms at
c1, and 798/1,054 ms vs 813/1,045 ms at c32. Quality is also unchanged: mu-bench-full WER 41.45% with
22.0% errors vs about 41.4% with 22.0%, and pipecat 3.68% with 1.3% errors vs about 3.8% with 1.3%.
Those error rates (about 1% of pipecat streaming sessions, 22% of mu-bench-full clips) are
pre-existing in the preset and the bench, and pod logs show no server-side errors.