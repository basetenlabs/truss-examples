This config was used to test throughput and concurrency characteristics of the Qwen 3.5-397B-A17B model (NVFP4) on a single B200:4 deployment to determine the replica count needed to sustain Valinor's 1,540 RPM target and complete the workload within the 72-hour window.

## AIperf command used
```shell
aiperf profile \
  --model "Qwen/Qwen3.5-397B-A17B" \
  --tokenizer "Qwen/Qwen3.5-397B-A17B" \
  --url $BASETEN_API_ENDPOINT \
  --endpoint-type chat \
  --api-key $BASETEN_API_KEY \
  --isl 1200 \
  --isl-stddev 600 \
  --osl 300 \
  --osl-stddev 450 \
  --prefix-prompt-length 28800 \
  --num-prefix-prompts 1 \
  --concurrency 64 \
  --warmup-request-count 5 \
  --benchmark-duration 180 \
  --random-seed 77
```

## peak throughput
- 7.9M tokens/m
- 4.35 req/s
- 261 RPM