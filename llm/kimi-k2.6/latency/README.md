# Kimi K2.6

This Truss serves Moonshot AI's [Kimi-K2.6](https://huggingface.co/moonshotai/Kimi-K2.6)
as an NVFP4 checkpoint through TRT-LLM (Dynamo with cache-aware routing) on
eight B200 GPUs. Kimi K2.6 is a mixture-of-experts model purpose-built for
agentic workflows and complex coding tasks, with a 262,144-token context
window, tool calling, and a thinking mode that is enabled by default.

## Configuration

| Setting | Value |
| --- | --- |
| Checkpoint | `baseten-admin/kimik26-final-nvfp4-v6` (NVFP4 quant of `moonshotai/Kimi-K2.6`) |
| Hardware | `B200:8` |
| Engine | TRT-LLM via Dynamo cache-aware routing, PyTorch backend |
| Context | 262,144 tokens |
| KV cache | FP8 with block reuse |
| Served model name | `moonshotai/Kimi-K2.6` (alias: `moonshotai/Kimi-K2.5`) |
| Endpoint | OpenAI-compatible `/v1/chat/completions` |
| Thinking | Enabled by default; dedicated thinking sampling params |
| Structured output | `xgrammar` guided decoding, structural tool choice |

## Usage

```python
from openai import OpenAI

client = OpenAI(
    api_key="<BASETEN_API_KEY>",
    base_url="https://model-<MODEL_ID>.api.baseten.co/environments/production/sync/v1",
)

response = client.chat.completions.create(
    model="moonshotai/Kimi-K2.6",
    messages=[
        {"role": "user", "content": "Write a Python function that merges two sorted lists."}
    ],
    max_tokens=1024,
)

print(response.choices[0].message.content)
```

## Sources and validation

- [Official model card](https://huggingface.co/moonshotai/Kimi-K2.6)

The model-registry PR deployment and smoke benchmark validate this B200
configuration before merge.