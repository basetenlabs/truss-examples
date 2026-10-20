# Nemotron 3.5 Embed 8B on vLLM

Serve [`nvidia/Nemotron-3.5-Embed-8B-BF16`](https://huggingface.co/nvidia/Nemotron-3.5-Embed-8B-BF16), NVIDIA's multimodal, multilingual retrieval embedding model, on one `H100` with vLLM. It embeds text queries and document pages given as text, an image, or both into one 4,096-dimensional, L2-normalized space. License: OpenMDW-1.1.

## Model

- **Architecture:** `Mistral3Model`: a Pixtral vision encoder (from Ministral-3-8B) and projector feeding the [Nemotron-3-Embed-8B](../../../Nemotron-3-Embed-8B/h100x1-bf16/throughput/README.md) language model, bidirectional (`is_causal: false`). 8.38B parameters, BF16.
- **Text embeddings match Nemotron-3-Embed-8B.** Only the vision encoder and projector were trained; the language model was kept frozen, so text retrieval accuracy and the embedding space are unchanged. The new capability is image retrieval.
- **Output:** mean-pooled, L2-normalized, 4,096 dims. Matryoshka truncation is supported (see [Smaller vectors](#smaller-vectors)).
- **Context:** 32,768 tokens, the limit on NVIDIA's model card, shared by text and image tokens. The checkpoint config allows 262,144, but NVIDIA trained and evaluated at 32,768.
- **Images:** resized server-side to 1,120 px on the longest edge by default; one image per input.

## Serving stack

- **`vllm/vllm-openai:v0.32.0`**, the version on NVIDIA's model card. `Mistral3Model` is native in vLLM, so there is no `--trust-remote-code`, and `config.json` sets mean pooling.
- **Weights** are pinned to revision `f3025617` and mounted through BDN. The repo is ungated, so no HF secret is declared.
- **Exact instance:** `resources.instance_type: H100` (one H100 80 GB), with no `accelerator`/`use_gpu` hints.
- **`startup_threshold_seconds`** allows for the image pull, weight mount, and `torch.compile` of the vision encoder (`compile_mm_encoder`, as on the model card).

### Why vLLM and not BEI or BIS

| Engine | Text | Images | Matryoshka | 32K context | Verdict |
|---|---|---|---|---|---|
| vLLM 0.32.0 | yes | yes | yes | yes | **Full featured** |
| BEI (`encoder`) | yes | no | yes | yes | Not full featured |
| BIS-LLM | no | no | no | no | Not applicable |

- **BEI** has no Mistral3/Pixtral multimodal adapter. The BEI path for this family aliases the text tower to a bidirectional Llama/Mistral encoder and drops the vision encoder. The result is a copy of Nemotron-3-Embed-8B (identical text embeddings) with no image retrieval.
- **BIS-LLM** serves text generation only. Baseten's engine matrix lists embedding and classification models as unsupported on BIS-LLM.

A text-only engine would remove the one capability this model adds over Nemotron-3-Embed-8B, so vLLM is the only full-featured option.

### Flags

| Flag | Why |
|---|---|
| `--runner pooling` | Embedding model; mean pooling and normalization come from the checkpoint. |
| `--hf-overrides.is_matryoshka true` | Lets the OpenAI `dimensions` parameter truncate and re-normalize, as the model card describes. |
| `--compilation-config.compile_mm_encoder true` | Compiles the Pixtral encoder, as on the model card. |
| `--chat-template-content-format openai` | Passes structured `messages` content (image and text blocks) to the checkpoint's chat template, as on the model card. |
| `--limit-mm-per-prompt.image 1` | One image per input, the format the model is trained on. A second image returns `400`. |
| `--max-model-len 32768` | Full published context. |
| `--max-num-batched-tokens 65536` | Must cover at least one full 32K input: vLLM does not chunk prefill for bidirectional models. |

**Prefix caching and chunked prefill** do not apply. Attention is bidirectional, so every token's state depends on the whole input and no prefix can be reused. vLLM turns both off for non-causal pooling models.

**No `vision_config.image_size` override.** The EA build ([#618](https://github.com/basetenlabs/model-registry/pull/618), vLLM 0.28) needed one: vLLM sized image placeholders from `image_size: 1540` while the processor resized to 1,120 px, so any larger image crashed the engine. vLLM 0.32 sizes placeholders from the processor's `longest_edge`, including per-request overrides, so the mismatch is gone. Keeping the checkpoint value also preserves the card's option to raise the resolution to 1,540 px.

## Prompt format

The model expects a retrieval prefix: `query:` for queries and `passage:` for documents. Its chat template renders the **system** message as the prefix, and the **user** message holds image and/or text blocks. It also escapes `[` and `<` in user text so a literal `[INST]` or `<s>` in a document isn't read as a control token. Send document text through `messages` or `/v2/embed`, not as plain `input` strings, so it goes through the template.

### `/v1/embeddings` with `messages` (OpenAI-compatible)

```python
import os
import requests

resp = requests.post(
    "https://model-xxxxxxxx.api.baseten.co/environments/production/sync/v1/embeddings",
    headers={"Authorization": f"Api-Key {os.environ['BASETEN_API_KEY']}"},
    json={
        "model": "nvidia/Nemotron-3.5-Embed-8B-BF16",
        "messages": [
            {"role": "system", "content": "passage:"},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "https://docs.nvidia.com/nemo-framework/user-guide/24.07/_images/nemo-llm-mm-stack.png"}},
                {"type": "text", "text": "NeMo Framework offers comprehensive functionalities for developing both LLMs and Multimodal Models."},
            ]},
        ],
        "encoding_format": "float",
    },
)
vec = resp.json()["data"][0]["embedding"]  # 4096 floats
```

One request returns one embedding. Image and text in the same message are embedded together.

### `/v2/embed`: the server applies the prefix from `input_type`

```python
resp = requests.post(
    "https://model-xxxxxxxx.api.baseten.co/environments/production/sync/v2/embed",
    headers={"Authorization": f"Api-Key {os.environ['BASETEN_API_KEY']}"},
    json={
        "model": "nvidia/Nemotron-3.5-Embed-8B-BF16",
        "input_type": "document",  # or "query"
        "inputs": [
            {"content": [{"type": "image_url", "image_url": {"url": "https://example.com/page.png"}}]},
            {"content": [{"type": "text", "text": "OCR text of another page"}]},
        ],
        "embedding_types": ["float"],
        "truncate": "END",
    },
)
vecs = resp.json()["embeddings"]["float"]
```

`/v2/embed` batches several inputs per request; this is the client NVIDIA's model card uses.

## Smaller vectors

Pass `dimensions` (for example `1024` or `512`) to `/v1/embeddings`. vLLM keeps the first N values and L2-normalizes them, which is the truncation NVIDIA documents. Retrieval quality drops as dimensions shrink, and faster for images than for text. NVIDIA doesn't publish per-dimension scores for this model; validate the size you pick on your own retrieval set.

## Higher image resolution

The card allows raising the longest edge from 1,120 to 1,540 px. Pass it per request with `"mm_processor_kwargs": {"size": {"longest_edge": 1540}}`. Larger images produce more image tokens and take longer to embed. Use the same resolution for every document in a corpus.

## Deploy

```bash
baseten model push
```