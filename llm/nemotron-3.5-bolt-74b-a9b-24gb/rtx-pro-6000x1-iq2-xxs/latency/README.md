# NVIDIA Nemotron 3.5 Bolt 74B-A9B-24GB — latency

Serves [NVIDIA's packed Bolt checkpoint](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB)
through vLLM's OpenAI-compatible `/v1/chat/completions` endpoint. The model has
74B total / 9B active parameters, a Mamba-2/Transformer LatentMoE backbone,
and a C-RADIO vision encoder. It accepts text, images, and video and emits text.
The checkpoint is ungated and licensed under
[OpenMDW-1.1](https://openmdw.ai/license/1-1/); no Hugging Face secret is required.

## Hardware and runtime

- Exact Baseten SKU: `resources.instance_type: RTX-PRO-6000` — one Blackwell
  RTX PRO 6000, 96 GiB VRAM, 16 vCPUs, and 116 GiB system RAM. No accelerator,
  CPU, or memory constraints are used to select a different instance implicitly.
- The vendor's `b12x` serving path requires SM120/SM121. Do not substitute an
  L4, H100, or B200 merely because the packed weights fit in memory.
- vLLM is pinned to the model card's required commit `77154ca6116ae34cbfe314beef1016bde9b63989`
  and Docker image digest `sha256:a7deff2c216def98c193352aca50fb03a2ee29231355b3033dc3789009f6d23e`.
  This new packed format needs the vendor-recommended nightly, not an older stable image.
  `b12x==1.5.1` and the vLLM V2 model runner are enabled.
  Kernel download runs on the GPU replica before `exec vllm serve`, not during
  the CPU-only image build, because this vLLM CLI initializes its device config.
- `b12x` describes its kernels as fast-moving and not production/datacenter-qualified.
  Treat this preset as experimental until registry CI and workload-specific testing pass.
- Weight revision: `c6cc2e8ffa353403213d6b9b18aaf8eb684d705a`, mounted at
  `/app/checkpoint/model`. The reasoning plugin and trusted remote model code come
  from that same pinned snapshot; there is no download from a moving branch at startup.
- Experts use IQ2-XXS (2.0625 effective bits); other layers mix NVFP4, Q8_0, and
  unquantized tensors. `iq2-xxs` names the expert format, not every tensor's dtype.
  vLLM detects ModelOpt mixed precision from the checkpoint. The safetensors
  shards load through `runai_streamer` from the Baseten weights mount.
- Context: 262,144 tokens, shared by prompt, media, history, and generated output.
  Tensor parallelism is one; vLLM active sequences and Baseten request concurrency
  are both capped at 32 to allow concurrent batching. This does not guarantee
  that 32 full-context requests fit simultaneously; capacity depends on token
  lengths and available cache memory. The 24GB name describes the compressed
  model, not the selected GPU's VRAM or a guarantee of 256K serving on a 24GB GPU.
- One image and one video per prompt; video sampling is 2 FPS, up to 256 frames,
  with video pruning at 0.5. Local-file media access is not enabled.
- Tool calls use `qwen3_coder`; reasoning uses the checkpoint's `ultra_v3` plugin.
  Set `chat_template_kwargs.enable_thinking` to enable or disable reasoning.
  `max_tokens` includes reasoning tokens, so allow a larger budget for thinking.

## LiLiCorr / DFlash speculative decoding

The preset enables NVIDIA's companion
[LiLiCorr drafter](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr)
with DFlash, seven speculative tokens, standard rejection sampling, and the
FlashInfer draft attention backend. The target verifies the proposed tokens;
the drafter is not a standalone served model. Standard rejection sampling is
used, not synthetic acceptance settings for performance simulation.

The ungated drafter is pinned to revision
`4c6a5613b6d887194a3a5493e3a18a5b3930bb86`, mounted separately at
`/app/checkpoint/drafter`, and licensed under OpenMDW-1.1. Both checkpoints load
local safetensors through `runai_streamer`; neither is fetched from a moving
Hugging Face branch at startup. Dotted speculative-config flags avoid shell
quoting ambiguity for nested JSON.

Concurrency remains capped at 32 at both layers, on the same exact
`RTX-PRO-6000` instance. The prior non-speculative smoke run
`run-8d5953c353bd` measured 266.67 aggregate output tokens/s for 1024 input /
1024 output tokens at client concurrency four. That result is not a DFlash
measurement; any speedup must be established by CI on the updated configuration.

## Example request

Use served model name `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB`:

```json
{
  "model": "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB",
  "messages": [{"role": "user", "content": "Explain binary search briefly."}],
  "max_tokens": 512,
  "temperature": 1.0,
  "top_p": 0.95,
  "chat_template_kwargs": {"enable_thinking": false},
  "stream": true
}
```

Registry PR CI owns deployment, smoke benchmarking, and cleanup. `results.json`
remains a missing-baseline placeholder until measured results are available.