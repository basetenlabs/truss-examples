# gpt-oss-120b (BIS-LLM)

This Truss serves OpenAI's [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b)
through the Baseten Inference Stack (TRT-LLM PyTorch backend behind Dynamo cache-aware
routing) on four B200 GPUs, tuned for throughput. It is the registry copy of the deployed
configuration from BIS Config Registry composer `gpt-oss-120b` (bcr_version 1.0.5).

The sibling `throughput` and `h100-throughput` presets in this directory serve the model
with vLLM (KNATIVE_V0); BIS-LLM requires its own preset because the backend rejects BIS
pushes into a knative model.

## Configuration

| Setting | Value |
| --- | --- |
| Checkpoint | `openai/gpt-oss-120b` at `b5c939de` (public, ungated, Apache-2.0) |
| Hardware | `B200:4`, tensor parallel 4, MoE expert parallel 4 |
| Engine | BIS-LLM `0.0.1-20260804033840-1d68536d`, TRT-LLM PyTorch backend, `TRTLLM` MoE backend, TRTLLM sampler |
| Context | 131,072 tokens (`max_seq_len`, `max_input_len`, `tokenizer_limit_length`) |
| Batching | `max_batch_size` 64, `max_num_tokens` 16,384, padded CUDA-graph batch sizes up to 64, chunked prefill, autotuner |
| KV cache | Block reuse, 120 GB host offload, 90% free-GPU-memory fraction |
| Served model name | `openai/gpt-oss-120b` |
| Processors | `harmony` chat processor; reasoning effort policy (`low`/`medium`/`high`, default `medium`) |
| Routing | Worker tree with WSPT queue policy, rate limiting, and cache-miss-aware routing |
| Endpoint | OpenAI-compatible `/v1/chat/completions` |

## Usage

```python
from openai import OpenAI

client = OpenAI(
    api_key="<BASETEN_API_KEY>",
    base_url="https://model-<MODEL_ID>.api.baseten.co/environments/production/sync/v1",
)

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
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