# Nemotron-3-Embed-8B-BF16 — vLLM Truss (HF weights, BDN-mounted)

A lean [`docker_server`](https://docs.baseten.co/) Truss that serves the GA build
[`nvidia/Nemotron-3-Embed-8B-BF16`](https://huggingface.co/nvidia/Nemotron-3-Embed-8B-BF16)
on the stock `vllm/vllm-openai` image. Weights are declared in a `weights:` block
pinned to a commit SHA, mirrored to Baseten's BDN at deploy time and pre-mounted to a
local path before the container starts — so vLLM loads from local disk and never calls
Hugging Face at runtime (no cold-start 429s). Nothing is baked into the image. Identical
in shape to the 1B truss; the only differences are the repo, the native embedding width
(**4096**), and the Matryoshka dimension list.

## Config highlights

- **Native `Ministral3Model`** — no `--trust-remote-code`.
- **No `--pooler-config`** — `config.json` ships `pooling: avg`; vLLM applies it.
- **No local `data/` dir** — weights come from the `weights:` block (HF source, pinned
  SHA), BDN-mirrored and pre-mounted for a small push and fast cold starts.
- **`--max-model-len 32768`** — the real context window (card notes eval used 4096).
- **`vllm/vllm-openai:v0.24.0`** — validated for `/v1/embeddings` (v0.25.0 needed only
  for the native `/v2/embed` endpoint).

## Endpoint

OpenAI-compatible `/v1/embeddings`. This path does **not** auto-apply the retrieval
prompts, so prefix inputs yourself: queries → `"query: <text>"`, documents →
`"passage: <text>"`.

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://model-xxxxxxxx.api.baseten.co/environments/production/sync/v1",
    api_key=os.getenv("BASETEN_API_KEY"),
)

resp = client.embeddings.create(
    model="nvidia/Nemotron-3-Embed-8B-BF16",
    input=["query: What is the capital of France?"],
)
print(len(resp.data[0].embedding))  # 4096 (native)
```

## Variable output dimensions (Matryoshka / MRL)

Native width is **4096**. Per NVIDIA, the family supports dynamic dimensions; this
truss enables them server-side with `--hf-overrides`, patching the Matryoshka metadata
in memory at load (no mirror repo, no `data/` override):

```
--hf-overrides "{\"is_matryoshka\":true,\"matryoshka_dimensions\":[512,1024,2048,4096]}"
```

```python
resp = client.embeddings.create(
    model="nvidia/Nemotron-3-Embed-8B-BF16",
    input=["query: What is the capital of France?"],
    dimensions=1024,         # must be one of matryoshka_dimensions
)
assert len(resp.data[0].embedding) == 1024
```

### ⚠️ Extra caveat vs. the 1B

The 8B model card, like the 1B's, does **not** document MRL — but unlike the 1B, we do
**not** have explicit out-of-band confirmation of the lossless **floor** for the 8B
(the "down to 512" figure was stated for the 1B). The `matryoshka_dimensions` list here
assumes the same behavior. **Validate quality at each dimension** (512/1024/2048 vs the
full 4096) on a real eval set before relying on the smaller sizes, and trim the list to
whatever actually holds up.

The generic caveats from the 1B truss also apply:

1. **Field names / vLLM version** — confirm v0.24.0 honors `is_matryoshka` /
   `matryoshka_dimensions` and that `dimensions=1024` returns a 1024-vector rather than
   erroring. If not, fall back to client-side truncation (slice `[:n]` then
   L2-renormalize — the model already emits normalized vectors).
2. **Allowed-list rejection** — vLLM typically rejects a `dimensions` value not in the
   list.

## Deploy

```bash
truss push
```