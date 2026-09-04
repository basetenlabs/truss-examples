Notes for FDE:

* This works on H100 or B200. But on H100 we need to offload text encoder to save space which affects throughput/latency.
* H100 can serve 1024x1024 images while B200 can go up to 4096x4096.
* SGLANG_WARMUP_SIZE_ON_INIT is crucial for reducing first generation time. The dimensions are not too important, but a good match to real traffic will accelerate the first request more