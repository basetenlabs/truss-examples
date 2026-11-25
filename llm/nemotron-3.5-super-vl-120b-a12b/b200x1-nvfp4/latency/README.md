# Nemotron 3.5 Super VL 120B-A12B NVFP4 — BIS-LLM B200:1 latency

This preset is a public-artifact port of the Model API production config for
`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B` (reference mirror:
`mapi/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B/config.yaml`). It serves
NVIDIA's 120B-total / 12B-active hybrid Mamba-MoE vision-language model with
BIS-LLM, an SGLang worker and Dynamo cache-aware routing on one B200, with MTP
(NEXTN) speculative decoding and image and video input. Once published, it
backs the `nemotron-3.5-super-vl` library listing. The plain-vLLM
`llm/nemotron-3.5-super-vl-120b-a12b/b200x1-fp8/latency` preset stays in the
tree as a separate recipe.

## What differs from the production config

Most `bis_llm` fields are kept verbatim: worker image, engine flags, page
size, attention backend, context length, memory fraction, max running
requests, hierarchical cache, Mamba cache settings, speculative decoding,
parsers, `chat_processor`, reasoning policy, sampling defaults, vision limits,
routing, autoscaling and the remaining environment variables. Two changes are
not artifact renames, so this preset does not reproduce the production setup
exactly: it runs on the platform-default BIS chart instead of production's
pinned one, and it drops the cache-directory overrides. Run alone,
D(1024,1024) decode on the 2026-11-22 default chart matches the pinned chart
(813.8 vs 837.3 output tok/s at concurrency 4). The first two rows below are
those changes; the rest only replace private artifacts or internal names:

| Field | Production | This preset | Why |
| --- | --- | --- | --- |
| `bis_llm.version` | `0.0.1-20261001221953-8f302211` | not set (platform default) | The platform now rejects chart pins older than its minimum build time (baseten#44233) |
| `SGLANG_CACHE_DIR`, `XDG_CACHE_HOME` | `/root/.cache` | removed | Lets the worker use the platform's startup cache location |
| `weights[].source` | Private Baseten early-access mirror | `hf://nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4@da5b2650…` | NVIDIA's public, ungated repo, revision-pinned |
| `weights[].auth_secret_name` | `hf_access_token` | removed | The public repo needs no HF token |
| `mount_location`, `model_path`, `model_path_for_tokenizer` | `-ea-` paths | `/models/nemotron-3.5-super-vl-nvfp4` | Drop the early-access mirror's name |
| `alias_served_model_names` | includes an internal production alias | internal alias removed | Not a customer-facing name |
| Production model and deployment IDs | in the header comment | removed | Internal identifiers |
| `model_name` | Production name | `model:nemotron-3.5-super-vl-120b-a12b-bis preset:latency` | Registry naming convention |
| `resources` | `accelerator: B200`, `use_gpu`, CPU and memory | `instance_type: B200` | Registry convention |
| `model_metadata` | not set | description, tags, example input | Library listing metadata |

## Pinned artifacts

| Artifact | Pin |
| --- | --- |
| Worker image | `baseten/dynamo-cache-aware-routing:akorte-nem35-video-master-1018--sglang-a6a8c560…@sha256:015e73d9…` (same as production) |
| BIS chart | Platform default (not pinned) |
| Checkpoint | `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4@da5b2650a835292c678994a1fa3fb653250fb508` |

Like other BIS presets in this registry, the worker image is a Baseten-built
image that the platform pulls. It is not an anonymously pullable public image.
BIS-LLM deployments require a Business or Startup workspace.

## Open questions

- **Worker image tag.** The production image tag is a personal development
  build (`akorte-nem35-video-master-1018`). Someone on BIS should confirm it is
  the image customers should run.
- **Checkpoint equivalence.** The public revision `da5b2650` is newer than the
  production early-access mirror and has not been hash-compared with it.
- **Not deployed.** This preset has not been deployed or benchmarked yet. PR
  CI's smoke workload is text-only, so image and video input need a manual
  check.

## Access and license prerequisites

The checkpoint is public and ungated, so no HF token or secret is needed.
Review NVIDIA's license on the
[model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4)
before use.

## Request behavior

The served model name is `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`.
Thinking is off by default; set `reasoning_effort` to `low`, `medium`, `high`,
`xhigh` or `max` to enable it (`high` when enabled without a level). Sampling
defaults are temperature 1.0, top-p 0.95 and 16,384 max tokens. A request may
carry up to 96 images and 12 videos (320 MB of media in total).

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://model-<model-id>.api.baseten.co/environments/production/sync/v1",
    api_key="<BASETEN_API_KEY>",
)
response = client.chat.completions.create(
    model="nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/page.png"}},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ],
    max_tokens=4096,
)
print(response.choices[0].message.content)
```