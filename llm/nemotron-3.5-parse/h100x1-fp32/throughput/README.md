# Nemotron 3.5 Parse on vLLM

Serve [`nvidia/Nemotron-3.5-Parse`](https://huggingface.co/nvidia/Nemotron-3.5-Parse), NVIDIA's document page parser, on one `H100` with vLLM. Given one page image, it returns the page's text in reading order with semantic layout classes and bounding boxes, with tables and charts rendered as markdown or LaTeX. License: OpenMDW-1.1; the tokenizer is MIT.

## Model

- **Architecture:** `RadioMBartPrefixForCausalLM`: a C-RADIOv4-SO400M vision encoder with dynamic NaViT preprocessing, a convolutional adapter, and a 10-layer mBART decoder that reads the visual sequence as a causal prefix. It includes one trained multi-token-prediction (MTP) head. 0.63B parameters.
- **Precision:** the checkpoint is stored in FP32, which is why the folder is `h100x1-fp32`. It is served in BF16 (`--dtype bfloat16`), NVIDIA's serving and evaluation contract.
- **Input:** one RGB page image and a task prompt. The recommended maximum resolution is 1,664 × 2,048 (W × H).
- **Context:** 16,384 tokens (`max_sequence_length`), the model's full context, shared by image and output tokens.

## Serving stack

NVIDIA's serving contract is **vLLM 0.26.0 plus the `nemotron-parse` package**. The package's `vllm.general_plugins` entry point registers the architecture and turns on the bundled one-token MTP proposer.

- **`vllm/vllm-openai:v0.26.0`**, with the package installed in `build_commands` from the model repo at the same pinned revision as the weights (`e7ea7924`). vLLM, torch 2.11.0, and torchvision 0.26.0 already match the package's pins; it pins transformers to 5.14.1, NVIDIA's validated version.
- **Weights** are mounted through BDN at that revision, without NVIDIA's `reference/` benchmark artifacts and `tests/`. The repo is ungated, so no HF secret is declared.
- **Exact instance:** `resources.instance_type: H100` (one H100 80 GB), NVIDIA's reference hardware, with no `accelerator`/`use_gpu` hints. The FA3 attention path and the published speed profile are H100-specific.
- **`startup_threshold_seconds`** allows for the image pull, weight mount, and CUDA graph capture.

The flags are NVIDIA's reference H100 throughput profile from `docs/serving.md`:

| Flag | Why |
|---|---|
| `--trust-remote-code` | The config and processor classes ship in the repo (`auto_map`). |
| `--max-model-len 16384` | Full context. |
| `--max-num-seqs 192`, `--max-num-batched-tokens 16384` | NVIDIA's H100 throughput profile. |
| `--attention-config.flash_attn_version 3` | FA3, which the plugin's MTP scheduler is validated with. |
| `--limit-mm-per-prompt.image 1` | One page per request. |
| `--chat-template <checkpoint>/chat_template.jinja` | Concatenates message content, so the task prompt reaches the decoder unchanged. |
| `--no-enable-prefix-caching` | As in NVIDIA's reference. |
| `--distributed-executor-backend uni` | Single-process executor, as in NVIDIA's reference. |

**MTP speculative decoding** is on by default through the plugin (one draft token). Startup logs show the speculative config; vLLM's `SpecDecoding metrics` lines report acceptance.

**Prefix caching** stays off, as in NVIDIA's reference. Each request's prefix is its own page image, followed by a few control tokens, so there is nothing to reuse across requests. The only cache hit would be re-parsing an identical page.

### Why vLLM and not BIS or BEI

- **BIS-LLM** runs certified TensorRT-LLM or vLLM images that cannot load this architecture. It is registered only by NVIDIA's out-of-tree vLLM plugin, and BIS doesn't install customer packages.
- **BEI** serves embedding and classification models, not generation.

## Call the model

Use the OpenAI chat completions API with one page image and the task prompt. Keep `skip_special_tokens: false`: the coordinate and class tokens are the output.

```python
import base64
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://model-xxxxxxxx.api.baseten.co/environments/production/sync/v1",
    api_key=os.environ["BASETEN_API_KEY"],
)

with open("page.png", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

resp = client.chat.completions.create(
    model="nvidia/Nemotron-3.5-Parse",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
        ],
    }],
    max_tokens=8744,
    temperature=0.0,
    extra_body={"top_k": 1, "repetition_penalty": 1.1, "skip_special_tokens": False},
)
print(resp.choices[0].message.content)
```

These are NVIDIA's generation settings: greedy decoding, repetition penalty 1.1, up to 8,744 new tokens.

### Postprocessing

The raw output interleaves class tokens, content-normalized box coordinates, and text. NVIDIA's helpers turn it into Markdown with HTML tables, plus structured blocks with pixel boxes:

```bash
pip install "beautifulsoup4==4.13.3"
hf download nvidia/Nemotron-3.5-Parse --include 'postprocessing.py' 'latex2html.py' 'nemotron_parse/**' --local-dir .
```

```python
from PIL import Image
from postprocessing import postprocess_document

image = Image.open("page.png").convert("RGB")
document = postprocess_document(resp.choices[0].message.content, image_size=image.size)
print(document["markdown"])
```

Box coordinates are normalized to the image the model saw. Scale x by its width and y by its height.

## Deploy

```bash
baseten model push
```