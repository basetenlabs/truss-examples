# DeepSeek V4 Flash

This Truss serves [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
through vLLM on four B200 GPUs with tensor and expert parallelism. DeepSeek V4
Flash is a 284B-parameter mixture-of-experts model (13B active) with a
1M-token context window, positioned as a lighter, faster alternative to
DeepSeek V4 Pro for routine and reasoning workloads.

## Configuration

| Setting | Value |
| --- | --- |
| Checkpoint | `deepseek-ai/DeepSeek-V4-Flash@main` (requires `hf_access_token`) |
| Hardware | `B200:4`, tensor parallel size 4 with expert parallelism |
| Engine | vLLM (`baseten/vllm-openai` image), DeepSeek V4 tool/reasoning parsers |
| Context | 1M tokens (`--max-model-len auto`) |
| KV cache | FP8, prefix caching enabled |
| MoE backend | `deep_gemm_mega_moe` with FP4 indexer cache |
| Speculative decoding | MTP, 3 speculative tokens |
| Served model name | `deepseek-ai/DeepSeek-V4-Flash` |
| Endpoint | OpenAI-compatible `/v1/chat/completions` |

## Usage

```python
from openai import OpenAI

client = OpenAI(
    api_key="<BASETEN_API_KEY>",
    base_url="https://model-<MODEL_ID>.api.baseten.co/environments/production/sync/v1",
)

response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V4-Flash",
    messages=[
        {"role": "user", "content": "Explain the CAP theorem in three concise sentences."}
    ],
    max_tokens=1024,
)

print(response.choices[0].message.content)
```

## Sources and validation

- [Official model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)

The model-registry PR deployment and smoke benchmark validate this B200
configuration before merge.