# EmbeddingGemma 2 on vLLM

Serve [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2), Google DeepMind's Apache-2.0 multimodal embedding model, with vLLM's OpenAI-compatible `/v1/embeddings` route. It maps text (including code), images, audio, video and interleaved combinations into one 768-dimensional, L2-normalized vector space. The checkpoint is 740M parameters: a 270M Gemma text model plus a 170M vision encoder and a 300M audio encoder, with an 8,192-token context shared across modalities.

The preset runs on one `H100` in bfloat16, with all three encoders loaded.

## Serving stack

vLLM added the `EmbeddingGemma2Model` architecture in [vllm-project/vllm#60254](https://github.com/vllm-project/vllm/pull/60254) on 2026-10-06, a day after v0.31.0 shipped, and no release or Docker image contains it yet. The config therefore starts from `vllm/vllm-openai:v0.31.0` and installs two packages in `build_commands`:

- the per-commit CUDA 13.0 vLLM wheel for main at `a8260cbc`, which includes the architecture and uses the same torch as the image;
- `transformers==5.19.0`, the first release that ships the `embedding_gemma2` module vLLM imports. Commit `a8260cbc` is the one that raised vLLM's transformers ceiling to admit it.

Once a vLLM release image includes `a8260cbc`, switch `base_image` to it and delete `build_commands`.

Weights come from the `weights:` block, pinned to revision `914f7f89`, mirrored to BDN and mounted before the container starts, so nothing calls Hugging Face at runtime. The repo is ungated, so no `hf_access_token` is declared.

vLLM selects its Triton attention backend for this model because four of its layers use a 512-wide head, and it samples video at 1 fps, up to 32 frames, matching the Hugging Face processor. Both happen without extra flags.

## Task prefixes

`/v1/embeddings` does not apply Sentence Transformers prompt names, so prefix text inputs yourself. Prefixes are optional but improve quality; media inputs take none. Use matching conventions on both sides of a comparison.

| Use | Query side | Document side |
|-----|-----------|---------------|
| Search / retrieval | `task: search result \| query: ` | `title: none \| text: ` |
| Question answering | `task: question answering \| query: ` | `title: none \| text: ` |
| Fact checking | `task: fact checking \| query: ` | `title: none \| text: ` |
| Code retrieval | `task: code retrieval \| query: ` | `title: none \| text: ` |
| Classification | `task: classification \| query: ` | same |
| Clustering | `task: clustering \| query: ` | same |
| Sentence similarity | `task: sentence similarity \| query: ` | same |

For a document with a title, replace `none` with the title: `title: <title> | text: <body>`.

## Call the model

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://model-xxxxxxxx.api.baseten.co/environments/production/sync/v1",
    api_key=os.environ["BASETEN_API_KEY"],
)

resp = client.embeddings.create(
    model="google/embeddinggemma-2",
    input=[
        "task: search result | query: What causes the northern lights?",
        "title: none | text: Auroras occur when charged particles from the Sun collide with gases in Earth's upper atmosphere.",
    ],
)
print(len(resp.data[0].embedding))  # 768
```

`model` must be `google/embeddinggemma-2`, the server's `--served-model-name`.

### Images, audio and video

Send multimodal inputs as chat-style `messages` to the same route. Each request returns one embedding for the whole message, so text and media in one message are embedded jointly.

```python
import os
import requests

resp = requests.post(
    "https://model-xxxxxxxx.api.baseten.co/environments/production/sync/v1/embeddings",
    headers={"Authorization": f"Api-Key {os.environ['BASETEN_API_KEY']}"},
    json={
        "model": "google/embeddinggemma-2",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"}},
                ],
            }
        ],
        "encoding_format": "float",
    },
)
print(len(resp.json()["data"][0]["embedding"]))  # 768
```

Audio uses `{"type": "audio_url", ...}` and video uses `{"type": "video_url", ...}`. Per the model card, an image costs about 280 tokens, a video frame about 140, and a second of audio 25, all counted against the 8,192-token context.

## Smaller vectors (Matryoshka)

The model is Matryoshka-trained at 768, 512, 256 and 128 dimensions. The checkpoint's `config.json` carries no Matryoshka metadata, so the config adds it at load time with `--hf-overrides '{"matryoshka_dimensions": [128, 256, 512, 768], "embedding_size": 768}'`. The `embedding_size` override is needed because vLLM checks `dimensions` against it, and without it vLLM falls back to the text model's hidden size, 512, and rejects `dimensions=768`. Pass `dimensions` to request a smaller vector; vLLM truncates and re-normalizes it server-side, and rejects sizes not in that list.

```python
resp = client.embeddings.create(
    model="google/embeddinggemma-2",
    input=["task: search result | query: What causes the northern lights?"],
    dimensions=256,
)
```

The card reports near-lossless quality down to 256 dimensions; at 128, multimodal quality degrades substantially. Queries and documents must use the same size.

## Precision

The preset serves in bfloat16 with `--dtype bfloat16`. The model card warns against float16: activations can exceed its range and silently produce NaN or degraded vectors.

## Deploy

```bash
baseten model push
```