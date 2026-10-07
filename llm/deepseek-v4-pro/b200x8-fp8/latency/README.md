# DeepSeek-V4-Pro (BIS-LLM)

This Truss serves DeepSeek's [DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
through the Baseten Inference Stack (TRT-LLM PyTorch backend behind Dynamo cache-aware
routing) on eight B200 GPUs with MTP speculative decoding. It is the registry copy of the
deployed configuration from BIS Config Registry composer `dsv4-pro-golden` (bcr_version 1.0.9).

This preset replaces the vLLM configuration of the same checkpoint, which published the
`deepseek-v4-pro-latency` listing.

## Configuration

| Setting | Value |
| --- | --- |
| Checkpoint | `deepseek-ai/DeepSeek-V4-Pro` at `5607980f` (public, ungated) |
| Hardware | `B200:8`, tensor parallel 8, MoE expert parallel 4 |
| Startup health-check window | 2,400 seconds (40 minutes) for checkpoint download and TRT-LLM initialization |
| Engine | BIS-LLM `0.0.1-20260601190849-691c46ce`, TRT-LLM PyTorch backend, `TRTLLM` MoE backend |
| Context | 1,048,576 tokens (`max_seq_len`, `max_input_len`, `tokenizer_limit_length`) |
| Speculative decoding | MTP, draft length 3, advanced sampling |
| KV cache | FP8, 128-token blocks, block reuse, 500 GB host offload with TP-MLA replicated host offload, 90% free-GPU-memory fraction |
| Batching | `max_batch_size` 8, `max_num_tokens` 16,384, chunked prefill |
| Served model name | `deepseek-ai/DeepSeek-V4-Pro` |
| Processors | `deepseek_v4` chat processor, reasoning parser, and tool-call parser; arguments as JSON |
| Thinking | Enabled by default; `tokenizer_max_new_tokens_limit` 32,768 |
| Sampling defaults | `temperature` 1, `top_p` 1 |
| Autoscaling | `in_flight_tokens` target 130,000 |
| Endpoint | OpenAI-compatible `/v1/chat/completions` |

`model_path_for_tokenizer` points at `tokenizer/` under the weights mount. That folder is
not part of the Hugging Face repository; BIS-LLM derives it as a sub-mount of the single
`weights[]` entry, as in the deployed listing.

## Usage

```python
from openai import OpenAI

client = OpenAI(
    api_key="<BASETEN_API_KEY>",
    base_url="https://model-<MODEL_ID>.api.baseten.co/environments/production/sync/v1",
)

response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V4-Pro",
    messages=[{"role": "user", "content": "What is the meaning of life?"}],
    max_tokens=1024,
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

## Validation

Registry PR CI deploys this configuration, calls its playground example, runs the LLM smoke
benchmark, and deactivates the temporary deployment. BIS-LLM deployments require a
workspace entitled to the referenced `gpuTRTImage`.