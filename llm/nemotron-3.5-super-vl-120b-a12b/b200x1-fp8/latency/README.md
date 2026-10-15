# Nemotron 3.5 Super VL 120B-A12B (NVFP4): latency preset

A [`docker_server`](https://docs.baseten.co/) Truss serving NVIDIA's
Nemotron-3.5-Super-VL-120B-A12B-NVFP4 with vLLM on a single B200. It accepts text, image
and video input through the OpenAI-compatible chat completions endpoint, and uses the
checkpoint's built-in MTP head for speculative decoding.

## Model

- **Architecture:** `NemotronH_Omni_Reasoning_V3`. A hybrid Mamba-2 + MoE + Attention
  language backbone (88 layers, 512 routed experts, 22 active) with a C-RADIO v4-H vision
  encoder and MLP projector. 120B total / 12B active parameters.
- **Quantization:** ModelOpt `MIXED_PRECISION` (`quant_method: modelopt_mixed`): NVFP4
  (group size 16) for the MoE experts, FP8 for the Mamba and attention projections, an FP8
  KV cache. The vision encoder and projector, embeddings, MoE router and latent
  projections, Mamba `conv1d`, and the MTP layer's attention stay unquantized. vLLM reads
  all of this from the checkpoint, so there's no `--quantization` flag. NVFP4 needs
  Blackwell GPUs.
- **MTP:** one MTP layer (`num_nextn_predict_layers: 1`), "MTP-boosted" in this release.
- **Context:** up to 1M positions in the backbone and tokenizer; this config caps at
  262144 tokens.
- **Reasoning:** on by default. Pass `chat_template_kwargs: {"enable_thinking": false}`
  for direct answers. Recommended sampling is `temperature: 1.0`, `top_p: 0.95`.

## Weights

Weights load from the public HF repo
[`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4)
(75 GB), pinned by commit SHA and mounted through BDN. No HF token is needed.

This release also updates `chat_template.jinja`: when an image or video follows text in
the same message, it now inserts a newline before the media tag. The previous template
ships alongside it as `chat_template_backup.jinja`; vLLM uses `chat_template.jinja`.

## Config highlights

- **`vllm/vllm-openai:v0.30.0`**: the first vLLM release that contains commit `2a02f6ef`,
  the nightly NVIDIA's model card validates. Don't pin a nightly tag: Docker Hub prunes them
  after a few weeks (both `nightly-2a02f6ef…` and `cu129-nightly-2a02f6ef…` are gone).
  v0.30.0 serves the architecture natively (`nano_nemotron_vl`), maps it to
  `nemotron_h_mtp` for speculative decoding, and loads `modelopt_mixed` checkpoints.
- **One `B200`, no tensor parallelism**: the card's minimum for NVFP4. 75 GB of weights on
  a 180 GB GPU leaves plenty of room for KV cache, MTP and video prefill.
- **MTP speculative decoding** (`--speculative-config.method mtp`, 3 speculative tokens).
  The card doesn't give a speculative config for vLLM, so 3 tokens is our choice.
- **Media flags follow the card's recommended recipe:** one image and one video per prompt
  (prevents OOMs), EVS video pruning at 0.5, and 2 FPS / 256 frames video sampling (the
  720p-and-below profile). Frame sampling is a server-launch setting, not a request
  parameter.
- **`--reasoning-parser nemotron_v3`, `--tool-call-parser qwen3_coder`**, as in the card.
- **`--mamba-ssm-cache-dtype float32`** matches `mamba_ssm_cache_dtype` in the checkpoint.

## Example

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://model-xxxxxxxx.api.baseten.co/environments/production/sync/v1",
    api_key="YOUR_BASETEN_API_KEY",
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
    temperature=1.0,
    top_p=0.95,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(response.choices[0].message.content)
```

PDFs aren't accepted directly: render each page to PNG and send it as an `image_url`.

## Deploy

```bash
baseten model push
```