# Nemotron-3-Embed-1B-BF16 — vLLM Truss (HF weights, BDN-mounted)

A lean [`docker_server`](https://docs.baseten.co/) Truss that serves the GA build
[`nvidia/Nemotron-3-Embed-1B-BF16`](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16)
on the stock `vllm/vllm-openai` image. Weights are declared in a `weights:` block
pinned to a commit SHA, mirrored to Baseten's BDN at deploy time and pre-mounted to a
local path before the container starts — so vLLM loads from local disk and never calls
Hugging Face at runtime (no cold-start 429s). Nothing is baked into the image.

## Why this differs from the early-access truss

The GA repo is a **native `Ministral3Model`** embedding architecture, unlike the
early-access `Nemotron3EmbeddingModel` build that needed `trust_remote_code` and a
bundled local checkpoint. Concretely:

- **No `--trust-remote-code`** — native arch, recognized by vLLM.
- **No `--pooler-config`** — `config.json` ships `pooling: avg`; vLLM applies it.
- **No local `data/` dir** — weights come from the `weights:` block (HF source, pinned
  SHA), BDN-mirrored and pre-mounted for a small push and fast cold starts.
- **`--max-model-len 32768`** — the real context window (card notes eval used 4096).

## Endpoint

OpenAI-compatible `/v1/embeddings`. This path does **not** auto-apply the retrieval
prompts, so prefix inputs yourself:

- queries → `"query: <text>"`
- documents → `"passage: <text>"`

(NVIDIA's native `/v2/embed` endpoint applies these automatically via an `input_type`
field, but it requires bumping the base image to `vllm/vllm-openai:v0.25.0`.)

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://model-xxxxxxxx.api.baseten.co/environments/production/sync/v1",
    api_key=os.getenv("BASETEN_API_KEY"),
)

resp = client.embeddings.create(
    model="nvidia/Nemotron-3-Embed-1B-BF16",
    input=[
        "query: What is the capital of France?",
        "passage: Paris is the capital of France.",
    ],
)
print(len(resp.data[0].embedding))  # 2048 (native)
# Note: `model` must match `--served-model-name` (the HF repo id).
```

## Variable output dimensions (Matryoshka / MRL)

The 1B model is **MRL-trained** and supports dynamic dimensions **down to 512,
near-losslessly** (per NVIDIA). The stock repo, however, ships **no** Matryoshka
metadata in `config.json`, so out of the box vLLM won't honor the OpenAI
`dimensions` parameter.

This truss patches that metadata **in memory at load time** with `--hf-overrides`
(no mirror repo, no `data/` override — weights are still the stock HF checkpoint):

```
--hf-overrides "{\"is_matryoshka\":true,\"matryoshka_dimensions\":[512,1024,2048]}"
```

With that in place, request a smaller vector via `dimensions` (slicing +
re-normalization happen server-side):

```python
resp = client.embeddings.create(
    model="nvidia/Nemotron-3-Embed-1B-BF16",
    input=["query: What is the capital of France?"],
    dimensions=512,          # must be one of matryoshka_dimensions
)
assert len(resp.data[0].embedding) == 512
```

### Caveats — verify on first deploy

1. **Field names / vLLM version.** `--hf-overrides` exists on the pinned v0.24.0, but
   the exact keys vLLM reads for Matryoshka have shifted across releases. Confirm a
   `dimensions=512` request returns a 512-vector rather than erroring. If the keys
   aren't recognized, fall back to **client-side truncation** (works with zero config):

   ```python
   import numpy as np
   full = np.array(resp.data[0].embedding)   # 2048-d, already L2-normalized
   v = full[:512]
   v = v / np.linalg.norm(v)                  # renormalize — required after slicing
   ```

2. **Allowed-list behavior.** With `matryoshka_dimensions` set, vLLM typically
   **rejects** a `dimensions` value not in the list. Add every size you'll use;
   drop ones you won't.

3. **Validate quality** at 512 vs 2048 on a small eval set — NVIDIA says near-lossless
   to 512, so this is confirmation, not exploration.

## Deploy

```bash
truss push
```