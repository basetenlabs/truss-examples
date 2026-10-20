# Nemotron 3.5 Rerank 8B on vLLM

Serve [`nvidia/Nemotron-3.5-Rerank-8B-BF16`](https://huggingface.co/nvidia/Nemotron-3.5-Rerank-8B-BF16), NVIDIA's multimodal, multilingual cross-encoder reranker, on one `H100` with vLLM. It scores a text query against a document that is text, a page image, or both, and returns a raw relevance logit: higher means more relevant. License: OpenMDW-1.1.

## Model

- **Architecture:** `Mistral3ForSequenceClassification`: a Pixtral vision encoder and a bidirectional Ministral3 language model (from Ministral-3-8B), mean-pooled into a one-output linear score head. 8.38B parameters, BF16.
- **Input:** one text query paired with one document; at most one image per document. Images are resized server-side to 1,120 px on the longest edge by default.
- **Context:** 32,768 tokens per query/document pair, the limit on NVIDIA's model card, shared by text and image tokens. The checkpoint config allows 262,144.
- **Output:** a raw logit (no sigmoid), as on the model card.

## Serving stack

- **`vllm/vllm-openai:v0.32.0`**, the version on NVIDIA's model card. The flags are NVIDIA's `vllm serve` command, except that `--max-num-seqs` is raised from 4 to 64 for a throughput preset.
- **Weights** are pinned to revision `39aa0152` and mounted through BDN. The repo is ungated, so no HF secret is declared.
- **Exact instance:** `resources.instance_type: H100` (one H100 80 GB), with no `accelerator`/`use_gpu` hints.
- **`startup_threshold_seconds`** allows for the image pull, weight mount, and `torch.compile` of the vision encoder.

### Why vLLM and not BEI or BIS

| Engine | Text pairs | Image documents | 32K context | Verdict |
|---|---|---|---|---|
| vLLM 0.32.0 | yes | yes | yes | **Full featured** |
| BEI (`encoder`) | yes, after a `-seq` conversion | no | yes | Not full featured |
| BIS-LLM | no | no | no | Not applicable |

- **BEI** builds causal and bidirectional text towers for sequence classification, but has no Mistral3/Pixtral multimodal adapter. A BEI build would score text pairs only.
- **BIS-LLM** serves text generation only. Baseten's engine matrix lists classification models as unsupported on BIS-LLM.

### Flags

| Flag | Why |
|---|---|
| `--runner pooling` | Sequence classification. |
| `--pooler-config.seq_pooling_type MEAN`, `--pooler-config.use_activation false` | Mean pooling into the score head; raw logits, as on the model card. |
| `--chat-template <checkpoint>/chat_template.jinja` | vLLM applies a score template only when one is passed explicitly. The template renders `query: …\n\npassage: …` (and the image placeholder), the training format, and escapes control-token strings in user text. |
| `--chat-template-content-format openai` | Passes structured `messages` content (image and text blocks) to the template. |
| `--compilation-config.compile_mm_encoder true` | Compiles the Pixtral encoder, as on the model card. |
| `--limit-mm-per-prompt.image 1` | At most one image per document; the template rejects more. |
| `--max-model-len 32768` | Full published context. |
| `--max-num-batched-tokens 65536` | Must cover at least one full 32K pair: vLLM does not chunk prefill for bidirectional models. |

**Prefix caching and chunked prefill** do not apply. Attention is bidirectional, so a shared query prefix can't be reused across documents. vLLM turns both off for non-causal pooling models.

## Call the model

### Text documents: `/v1/rerank` (default route)

```python
import os
import requests

resp = requests.post(
    "https://model-xxxxxxxx.api.baseten.co/environments/production/sync/v1/rerank",
    headers={"Authorization": f"Api-Key {os.environ['BASETEN_API_KEY']}"},
    json={
        "model": "nvidia/Nemotron-3.5-Rerank-8B-BF16",
        "query": "What is NVIDIA NeMo Framework?",
        "documents": [
            "NeMo Framework offers comprehensive functionalities for developing both LLMs and Multimodal Models.",
            "A biological foundation model designed to analyze and generate DNA, RNA, and protein sequences.",
        ],
    },
)
for r in resp.json()["results"]:
    print(r["index"], r["relevance_score"])
```

With the checkpoint template set, vLLM renders each pair as `[{"role": "query"}, {"role": "document"}]` through it. That is the same prompt `/classify` builds below, and the tokenizer adds no BOS/EOS.

### Image documents: `/classify`

NVIDIA warns that the `/score` and `/rerank` paths can mishandle this checkpoint's image placeholder, so send documents with an image to `/classify`, one query/document pair per request:

```python
resp = requests.post(
    "https://model-xxxxxxxx.api.baseten.co/environments/production/sync/classify",
    headers={"Authorization": f"Api-Key {os.environ['BASETEN_API_KEY']}"},
    json={
        "model": "nvidia/Nemotron-3.5-Rerank-8B-BF16",
        "messages": [
            {"role": "query", "content": [{"type": "text", "text": "What is NVIDIA NeMo Framework?"}]},
            {"role": "document", "content": [
                {"type": "text", "text": "NeMo Framework offers comprehensive functionalities for developing both LLMs and Multimodal Models."},
                {"type": "image_url", "image_url": {"url": "https://docs.nvidia.com/nemo-framework/user-guide/24.07/_images/nemo-llm-mm-stack.png"}},
            ]},
        ],
        "add_special_tokens": False,
        "use_activation": False,
    },
)
score = resp.json()["data"][0]["probs"][0]  # raw logit
```

The document may be text, an image, or both. Send pairs concurrently to rank several documents. Image URLs must be reachable from the server; base64 data URLs also work.

## Deploy

```bash
baseten model push
```