# Ultramarine 1 Flash (FP8), 4x B200, SGLang, MTP speculative decoding

[paperinstruments/Ultramarine-1-Flash](https://huggingface.co/paperinstruments/Ultramarine-1-Flash)
(Paper Instruments' fine-tune of GLM-5.3-Flash, 320B total / 18B active, FP8 with BF16 shared
experts) on four B200s with SGLang v0.5.22, served as `paperinstruments/Ultramarine-1-Flash`
behind an OpenAI-compatible `/v1/chat/completions`.

## Source and differences

The start command is Paper Instruments' recommended SGLang command from the model card
("Serve Ultramarine 1 Flash Locally"), which they tested on 4x B300:

```
python -m sglang.launch_server --model-path paperinstruments/Ultramarine-1-Flash --tp-size 4 \
  --disable-shared-experts-fusion --context-length 262144 \
  --reasoning-parser glm45 --tool-call-parser glm47 --speculative-algorithm EAGLE
```

Differences:

- Runs on 4x B200 instead of 4x B300.
- Weights are pinned to `4dd4031735f044b7cdd050b66a15d6ee4e259111` and mounted at
  `/models/ultramarine-1-flash` instead of downloaded at start. Later commits change only the
  README and NOTICE.
- The image is pinned by digest (`lmsysorg/sglang:v0.5.22@sha256:e3a4f060…`).
- `--max-running-requests 48` is set explicitly. SGLang resets to 48 for speculative decoding
  anyway; `predict_concurrency` matches it.
- `--served-model-name`, `--host` and `--port` are set for Truss.

Everything else is SGLang's Blackwell default for this architecture. The benchmarked boot
resolved:

- EAGLE speculative decoding on the model's built-in MTP layer (3 steps, top-k 1, 4 draft
  tokens);
- FP8 weights and `fp8_e4m3` KV cache;
- TRT-LLM DSA attention, the `flashinfer_trtllm` MoE runner and Triton linear-attention decode;
- chunked prefill 16384, memory fraction 0.8037;
- about 6.5M KV tokens.

No reasoning-effort default is set, so the chat template's default (`max`) applies; clients set
`chat_template_kwargs: {reasoning_effort: low|high}` per request.

## Access

The Hugging Face repo is gated with automatic approval: accept the license on the model page,
then store a token that has access as the `hf_access_token` secret. Model providers need a
separate agreement with Paper Instruments (see the license).

## Validation

Measured on 4x B200 through b10-deploy batch bench, 2026-10-29, with this command before the
digest pin, secret rename and explicit `--max-running-requests`:

- Perf (48k input / 3k output tokens, 40% cached, single replica, 0 errors; job
  `bj-05d43d133fdd`): TPOT p50 3.0 / 4.8 / 9.6 / 13.7 ms at concurrency 1 / 4 / 16 / 32,
  2,011 output tok/s at 32.
- Quality (job `bj-ce32e86cd4f7`, temperature 0, effort `max`): IFEval 87.6% (541 prompts),
  BFCL simple_python 96.25%, multiple 94.0%, parallel 91.0%.
- Teacher-forced input logprobs against a second same-weights SGLang deployment (no spec
  decoding, bf16 SSM state) were indistinguishable from same-server run-to-run noise at 4k,
  35k, 139k and 253k tokens.

Not run: the full-length 1M context, image and video input (the model card lists
`image-text-to-text`; all checks above were text-only), and B300.