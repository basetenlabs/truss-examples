# Ling 3.1 Flash — B200 FP8 latency preset

Serve inclusionAI's public Ling 3.1 Flash checkpoint on **eight B200 GPUs**
with vLLM, TP8/EP8, and native three-token MTP speculative decoding.
The checkpoint is public and ungated; **no Hugging Face token is required**.

The vendor model card reports a 560B-parameter hybrid MLA/KDA mixture of
experts, with 25B active parameters per token, 512 routed experts (8 selected),
one shared expert, 63 KDA layers, and 9 gated MLA layers. Weight quantization
is FP8 E4M3 with 128×128 block scaling and dynamic activation quantization;
non-quantized computation is BF16. The checkpoint license metadata is MIT.

## Serving contract

| Setting | Value |
| --- | --- |
| Served model | `inclusionAI/Ling-3.1-flash` |
| Exact hardware | `instance_type: B200:8` — single node |
| Parallelism | TP8 with expert parallelism across the eight ranks |
| Engine | vLLM 0.30.0, pinned to the tested Docker manifest digest |
| MoE backend | Explicit `triton` |
| Speculative decoding | Native MTP, 3 draft tokens; no external drafter |
| Hybrid caching | Prefix caching with `mamba-cache-mode: align` |
| Configured context ceiling | 262,144 tokens, including prompt and requested output |
| Precision | Serialized block FP8; excluded modules and MTP gate use BF16 |
| Chat endpoint | OpenAI-compatible `/v1/chat/completions` on port 8000 |
| Health endpoints | `/health` |
| Parsers | `ling3` reasoning and tool-call parsers |
| Concurrency setting | 32; smoke-tested at concurrency 1 and 4 |
| Weight delivery | Public BDN mount with `runai_streamer` |
| Startup probe window | 3,000 seconds (50 minutes) |

**Context limit:** the public model card advertises 1M-token context, but the
pinned `config.json` declares 262,144 positions. This preset intentionally
retains the latter limit. It does not claim validated 1M-token serving or
certification at the maximum context length.

## Reproducible dependencies

- **Checkpoint:** `inclusionAI/Ling-3.1-flash` at
  `7b864daa2cedb1404e12bc52899def2d151a9719`.
- **Image:** `vllm/vllm-openai:v0.30.0` at Docker manifest digest
  `sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`.
- **Compatibility patch:** `packages/patch_ling_mtp_gate.py`, guarded by the
  exact vLLM version, upstream source SHA, and checkpoint shape/dtype contract.

Do not update the engine version, image digest, or checkpoint revision without
revalidating the compatibility patch and rerunning startup and functional
checks. The source guard deliberately fails rather than silently patching a
changed upstream implementation.

## Deploy

Install and authenticate the Baseten/Truss CLI, then deploy from this directory:

```bash
baseten model push
```

Retain the returned model and deployment IDs for the examples below. Initial
weight delivery and GPU kernel compilation can take many minutes. CI
deployments took roughly 22–35 minutes from creation to readiness, including
provisioning/loading/warm-up; this is not a startup SLA. Eight B200s
provide 1,440 GiB aggregate VRAM; the 54 serialized safetensors shards total
573,781,078,249 bytes (~574 GB / 534 GiB), which is not a measured runtime
footprint. Workload-specific KV cache and concurrency still need headroom.

## Chat and thinking mode

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["BASETEN_API_KEY"],
    base_url="https://model-<MODEL_ID>.api.baseten.co/deployment/<DEPLOYMENT_ID>/sync/v1",
)
response = client.chat.completions.create(
    model="inclusionAI/Ling-3.1-flash",
    messages=[{"role": "user", "content": "What is 15% of 240?"}],
    temperature=1.0,
    top_p=0.95,
    max_tokens=16384,
    extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": True}},
)
message = response.choices[0].message
print(getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None))
print(message.content)
```

Thinking is selected **per request**. Set `enable_thinking: false` for a direct
answer, or `stream=True` for streaming responses. The `ling3` parser separates
reasoning from the final answer; SDK/server versions may expose it as
`reasoning` or `reasoning_content`. The vendor recommends `temperature=1.0`,
`top_p=0.95`, and `top_k=20`. Keep prompt plus output inside the configured limit.

## Tool calling

Supply OpenAI-format tools with `tool_choice="auto"`:

```python
response = client.chat.completions.create(
    model="inclusionAI/Ling-3.1-flash",
    messages=[{"role": "user", "content": "What is the current temperature in Paris?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_temperature",
            "description": "Read the current temperature for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }],
    tool_choice="auto",
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(response.choices[0].message.tool_calls)
```

Your application executes the selected tool. Append the assistant message and
one `role: "tool"` message per call with its `tool_call_id`, then request the
final answer. The model does not execute external tools itself.

## Native MTP and compatibility details

The checkpoint includes one MTP module under `model.layers.72`. vLLM reuses
that module autoregressively for three draft positions. Three speculative
tokens are not a promise of three accepted tokens or a 3× speedup; acceptance
and throughput depend on the workload. No Ling 3.0 DSpark drafter is used.

The public checkpoint excludes its narrow target attention gates from block
FP8, but the MTP gate remains serialized as FP8 `[64,4096]` with F32 block
scales `[1,32]`. At TP8, an 8-row output partition is incompatible with
vLLM 0.30.0's 128-row block-FP8 allocation. The build-time patch restores
**only that draft gate** to BF16 using its original stored scales. It handles
either tensor/scale stream order and leaves target weights and all other
draft tensors unchanged. Unexpected source, shapes/dtypes, missing tensors,
or duplicates fail closed. No checkpoint is rewritten.

The Triton MoE backend avoids another B200 compatibility limitation: the
auto-selected TRT-LLM fused router permits at most 32 experts per group,
whereas this model has 512/8 = 64. These compatibility choices prioritize a
working, reproducible recipe; they are not claims of optimal backend performance.

## Validation and performance scope

The native-MTP recipe passed registry CI deployment, smoke benchmarking, and
cleanup on October 14, 2026. The smoke run reported **0% errors** in all cells:

| Input/output tokens | Concurrency | Output tokens/s | Mean TPOT |
| --- | --- | --- | --- |
| 128/128 | 1 | 95.77 | 8.61 ms |
| 128/128 | 4 | 251.03 | 13.93 ms |
| 1024/1024 | 1 | 103.64 | 9.26 ms |
| 1024/1024 | 4 | 254.09 | 15.00 ms |

Against one non-MTP control run on the same hardware/settings, observed output
throughput increased 36.9%/14.0% for 128/128 at concurrency 1/4, and 39.6%/10.2%
for 1024/1024. This is a **single smoke comparison**, not a noise-calibrated
production baseline or peak-throughput claim. The benchmark platform retried
a transient submission timeout; the reported run completed successfully.

**Functional release checks:** passed on October 14, 2026: direct answers, separated reasoning, SSE streaming, automatic tool selection and a mock tool-result round trip. A single capacity probe with **260,056 input tokens** and a 32-token output budget also completed successfully; this is not a retrieval-accuracy or concurrent-load certification.

Six CPU regression tests cover the MTP gate patch. `packages/check_release.py`
provides focused endpoint checks for direct answers, separated reasoning,
SSE streaming, automatic tool selection, and a mock tool-result round trip:

```bash
# Set BASETEN_API_KEY securely in your environment; do not commit its value.
python packages/check_release.py \
  --model-id <MODEL_ID> \
  --deployment-id <DEPLOYMENT_ID> \
  --report /tmp/ling-release-checks.json
```

Run checks only after the deployment is ACTIVE. Add `--long-context` to send
one ~260K-token capacity probe; this may take longer and is not a load test.
The default check script uses small fixed prompts; it is not a load benchmark or a
statistical accuracy evaluation. Maximum-length requests, concurrency 32,
broader code/reasoning workloads, and production SLOs are **not certified** by
these smoke checks. `results.json` retains candidate measurements with
`accepted: false`; that does not prevent public recipe downloads, but it is
not an accepted production/Explore performance baseline.

## References

- [Pinned public model card](https://huggingface.co/inclusionAI/Ling-3.1-flash/blob/7b864daa2cedb1404e12bc52899def2d151a9719/README.md)
- [Pinned checkpoint configuration](https://huggingface.co/inclusionAI/Ling-3.1-flash/blob/7b864daa2cedb1404e12bc52899def2d151a9719/config.json)
- [vLLM 0.30.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.30.0)
- [Bailing MTP implementation](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/models/bailing_moe_v3_mtp.py)