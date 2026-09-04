# MiMo-V2.5

This Truss serves [MiMo-V2.5](https://huggingface.co/XiaomiMiMo/MiMo-V2.5) through
vLLM on four B200 GPUs with tensor parallelism. MiMo-V2.5 is Xiaomi's native
omnimodal mixture-of-experts model — 310B total parameters, 15B active per token,
with a 1,048,576-token context window and native FP8 (block-wise e4m3) weights —
supporting text, image, video, and audio understanding in a unified architecture.

## Configuration

| Setting | Value |
| --- | --- |
| Checkpoint | `XiaomiMiMo/MiMo-V2.5@63651580ca774f8504f676040460aed3e1244ac1` (ungated; MIT) |
| Hardware | `B200:4`, tensor parallel size 4 (TP=8 hits an attention-projection shape mismatch with this FP8 checkpoint) |
| Engine | vLLM via the pre-built `vllm/vllm-openai:mimov25-cu129` image; stable vLLM does not support `mimo_v2` yet |
| Context | 1,048,576 tokens (`--max-model-len auto`) |
| Parsers | MiMo tool-call and reasoning parsers, auto tool choice |
| Speculative decoding | MTP, 1 speculative token (the only value vLLM's MiMo-V2 MTP implementation supports) |
| Multimodal | text, image, video, audio (`--limit-mm-per-prompt.image 1`) |
| Served model name | `XiaomiMiMo/MiMo-V2.5` |
| Endpoint | OpenAI-compatible `/v1/chat/completions` |

Thinking mode is enabled by default. Pass
`"chat_template_kwargs": {"enable_thinking": false}` in the request to disable it.

## Usage

```python
from openai import OpenAI

client = OpenAI(
    api_key="<BASETEN_API_KEY>",
    base_url="https://model-<MODEL_ID>.api.baseten.co/environments/production/sync/v1",
)

response = client.chat.completions.create(
    model="XiaomiMiMo/MiMo-V2.5",
    messages=[
        {"role": "user", "content": "Explain the CAP theorem in three concise sentences."}
    ],
    max_tokens=1024,
)

print(response.choices[0].message.content)
```

## Sources and validation

- [Official model card](https://huggingface.co/XiaomiMiMo/MiMo-V2.5)
- [vLLM recipe for MiMo-V2.5](https://recipes.vllm.ai/XiaomiMiMo/MiMo-V2.5) (author-verified on 4x H200; this config uses TP=4 on 4x B200)

The model-registry PR deployment and smoke benchmark validate this B200
configuration before merge.