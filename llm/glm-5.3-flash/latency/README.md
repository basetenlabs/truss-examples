# GLM-5.3 Flash

This Truss serves Z.ai's official native-FP8
[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) checkpoint through
vLLM on eight H100 GPUs. GLM-5.3-Flash is a multimodal mixture-of-experts model
with approximately 321B total parameters and 18B active parameters per token. It
supports text, image, and video input; reasoning; tool calling; and a native
1,048,576-token context window.

## Configuration

| Setting | Value |
| --- | --- |
| Checkpoint | `zai-org/GLM-5.3-Flash@3f1971b7b5f7a528c9c4ef6212c8785298a8c24a` |
| Architecture | `Glm5NextForConditionalGeneration`, 321B total / 18B active MoE |
| Precision | Native FP8 weights; BF16 KV cache on Hopper |
| Hardware | `H100:8`, tensor parallel size 8 |
| Context | 1,048,576 tokens |
| Maximum concurrent sequences | 16 |
| Server | Dedicated `vllm/vllm-openai:glm53-flash` image, digest-pinned in `config.yaml` |
| Endpoint | OpenAI-compatible `/v1/chat/completions` |
| Served model name | `zai-org/GLM-5.3-Flash` |
| Speculative decoding | Built-in MTP head, 5 speculative tokens |
| Tool parser | `glm47` with automatic tool choice |
| Reasoning parser | `glm45` |
| Default reasoning effort | `high` (request-level `reasoning_effort` overrides it) |
| Multimodal limits | One image and one video per prompt |

The dedicated vLLM image is required because GLM-5.3-Flash support has not yet
landed in a stable public vLLM image. H100 is a Hopper GPU, and the current model
implementation does not support FP8 KV cache on Hopper; the server therefore
keeps the checkpoint weights in FP8 but uses BF16 for KV cache. FP8 KV cache must
not be enabled for this deployment.

## Usage

```python
from openai import OpenAI

client = OpenAI(
    api_key="<BASETEN_API_KEY>",
    base_url="https://model-<MODEL_ID>.api.baseten.co/environments/production/sync/v1",
)

response = client.chat.completions.create(
    model="zai-org/GLM-5.3-Flash",
    messages=[
        {"role": "user", "content": "Explain sparse attention in three concise sentences."}
    ],
    temperature=1.0,
    top_p=0.95,
    max_tokens=512,
)

print(response.choices[0].message.content)
```

Requests that omit `reasoning_effort` use `high`. Clients can override the
server default per request with any effort supported by the checkpoint, such as
`low`, `high`, or `max`.

## Sources and validation

- [Official model card](https://huggingface.co/zai-org/GLM-5.3-Flash)
- [Official vLLM recipe](https://recipes.vllm.ai/zai-org/GLM-5.3-Flash)
- [GLM-5 technical report](https://arxiv.org/abs/2602.15763)

The model-registry PR deployment and smoke benchmark validate this H100
configuration before merge.