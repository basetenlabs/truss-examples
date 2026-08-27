# DeepSeek V4 Pro

This Truss serves [DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
through vLLM on eight B200 GPUs with tensor and expert parallelism. DeepSeek V4
Pro is a 1.6T-parameter mixture-of-experts model with a 1,048,576-token context
window, built for agentic workflows, complex coding, and multi-step reasoning.

## Configuration

| Setting | Value |
| --- | --- |
| Checkpoint | `deepseek-ai/DeepSeek-V4-Pro@main` (gated; requires `hf_access_token`) |
| Hardware | `B200:8`, tensor parallel size 8 with expert parallelism |
| Engine | vLLM (`baseten/vllm-openai` image), DeepSeek V4 tool/reasoning parsers |
| Context | 1,048,576 tokens (`--max-model-len auto`) |
| KV cache | FP8, prefix caching enabled |
| MoE backend | `deep_gemm_mega_moe` with FP4 indexer cache |
| Speculative decoding | MTP, 3 speculative tokens |
| Served model name | `deepseek-ai/DeepSeek-V4-Pro` |
| Endpoint | OpenAI-compatible `/v1/chat/completions` |

## Usage

```python
from openai import OpenAI

client = OpenAI(
    api_key="<BASETEN_API_KEY>",
    base_url="https://model-<MODEL_ID>.api.baseten.co/environments/production/sync/v1",
)

response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V4-Pro",
    messages=[
        {"role": "user", "content": "Explain the CAP theorem in three concise sentences."}
    ],
    max_tokens=1024,
)

print(response.choices[0].message.content)
```

## Sources and validation

- [Official model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)

The model-registry PR deployment and smoke benchmark validate this B200
configuration before merge.