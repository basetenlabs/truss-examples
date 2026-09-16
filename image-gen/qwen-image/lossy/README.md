# Qwen Image (lossy preset)

Serves [Qwen/Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512) (Apache-2.0) with
SGLang from the prebuilt image `baseten/qwen-image-2512-h100:v1.2`, with cache-dit lossy
optimization enabled. Weights are pinned to HF revision `25468b98e3276ca6700de15c6628e51b7de54a26`
and mounted from the Baseten Delivery Network.

- Hardware: `H100` (B200 also works; see notes below).
- Endpoint: `POST /v1/images/generations` (OpenAI images API shape), port 8000.
- Health: `/health_generate`.
- Example request:

```json
{
  "prompt": "A cat holding up a sign that reads 'Hello, world!'",
  "size": "1024x1024",
  "guidance_scale": 1.0,
  "num_inference_steps": 25,
  "response_format": "b64_json"
}
```

Publishes to the Model Library listing `qwen-image-throughput`.

Notes for FDE:

* This works on H100 or B200.
* H100 can serve 2048x2048 images while B200 can go up to 4096x4096.
* SGLANG_WARMUP_SIZE_ON_INIT is crucial for reducing first generation time. The dimensions are not too important, but a good match to real traffic will accelerate the first request more
* With lossy optimization, you can expect 2x speedup with very little perceptual loss.