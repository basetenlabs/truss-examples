# Qwen3.8 27B FP8

This Truss serves the official block-scaled FP8 checkpoint of
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) through vLLM on one
H100 GPU. Qwen3.8-27B is a dense multimodal model with 64 language-model layers:
16 full-attention layers and 48 linear-attention layers. It supports a native
262,144-token context window, vision input, reasoning, tool calling, and a built-in
multi-token-prediction draft head.

## Configuration

| Setting | Value |
| --- | --- |
| Checkpoint | `Qwen/Qwen3.8-27B-FP8@017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` |
| Precision | Block-scaled FP8 weights; FP8 KV cache |
| Hardware | `H100:1` |
| Context | 262,144 tokens |
| Maximum concurrent sequences | 64 |
| Server | `vllm/vllm-openai:qwen38` (digest pinned in `config.yaml`) |
| Endpoint | OpenAI-compatible `/v1/chat/completions` |
| Served model name | `Qwen/Qwen3.8-27B` |
| Speculative decoding | Built-in MTP head, 3 speculative tokens |
| KV offload | `SimpleCPUOffloadConnector`, 64 GiB host memory per rank |
| Vision encoder | Data-parallel mode; one image per prompt |
| Tool parser | `qwen3_coder` |
| Reasoning parser | `qwen3` |

The `qwen38` vLLM image is required for this initial configuration because the
gated-delta-network speculative-decoding fixes used by Qwen3.8 are not yet present in a
released vLLM tag. The image is pinned to its multi-architecture digest for reproducibility.
The recipe UI's generic Simple-offload preset reserves 220 GiB of host memory per rank;
this Truss uses 64 GiB so it fits inside the allocatable host memory of the harmonized
`H100` SKU, which `resources.instance_type` pins explicitly.

## Usage

```python
from openai import OpenAI

client = OpenAI(
    api_key="<BASETEN_API_KEY>",
    base_url="https://model-<MODEL_ID>.api.baseten.co/environments/production/sync/v1",
)

response = client.chat.completions.create(
    model="Qwen/Qwen3.8-27B",
    messages=[
        {"role": "user", "content": "Give me three prime numbers greater than 100."}
    ],
    temperature=1.0,
    top_p=0.95,
    max_tokens=512,
)

print(response.choices[0].message.content)
```

The checkpoint supports per-request thinking controls through
`chat_template_kwargs`, including `{"enable_thinking": false}` and adaptive reasoning via
`{"reasoning_effort": "low"}` (`low`, `medium`, or `xhigh`).

## Sources and validation

- [Selected vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B?hardware=h100&variant=fp8&kv_offload=simple&features=tool_calling%2Creasoning%2Cencoder_parallel%2Cspec_decoding)
- [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B)
- [Official FP8 checkpoint](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)

The recipe was added on 2026-08-14 and reports end-to-end verification on GB300. This H100
configuration is validated by the model-registry PR deployment and smoke benchmark before
merge.