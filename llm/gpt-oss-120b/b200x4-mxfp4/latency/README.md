# gpt-oss-120b (BIS-LLM, Model API config)

This Truss serves OpenAI's [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b)
through the Baseten Inference Stack (TRT-LLM PyTorch backend behind B10 cache-aware
routing) on four B200 GPUs. It is a public-artifact port of the config serving the
`openai/gpt-oss-120b` Model API in production
([`mapi/openai/gpt-oss-120b`](../../../../mapi/openai/gpt-oss-120b/config.yaml)).

Every `bis_llm` setting matches production, including the serving image
(`gpuTRTImage: baseten/bis-llm-trtllm:green-mp--ec11764bde-4884584f6-49e724f969`) and BIS
version `0.0.1-20261001221953-8f302211`. The image pin is required: the platform-default
image that `engine_backend: trtllm` alone selects does not accept
`chat_processor: harmony` (its frontend fails `RawConfig` validation at startup). Like
other BIS presets in this registry, the image is pulled by the platform and is not an
anonymously pullable public image. The checkpoint is public and ungated, so no Hugging
Face secret is needed.

The [`throughput`](../throughput/) preset on the same hardware is the earlier BIS Config
Registry composer config (MoE expert parallel 4, 16,384 max tokens per batch).

Deploy-only: this preset has no `library_id`, so it is not listed in the Model Library. The
[`h100x2-mxfp4/throughput`](../../h100x2-mxfp4/throughput/) preset owns the `gpt-oss-120b` listing.

## Configuration

| Setting | Value |
| --- | --- |
| Checkpoint | `openai/gpt-oss-120b` at `b5c939de` (public, ungated, Apache-2.0) |
| Hardware | `B200:4`, tensor parallel 4, MoE expert parallel 2 |
| Engine | BIS-LLM, production `bis-llm-trtllm:green-mp` image, BIS `0.0.1-20261001221953-8f302211`, TRT-LLM PyTorch backend, `TRTLLM` MoE backend, TRTLLM sampler |
| Context | 131,072 tokens (`max_seq_len`, `max_input_len`, `tokenizer_limit_length`) |
| Batching | `max_batch_size` 64, `max_num_tokens` 32,768, padded CUDA-graph batch sizes up to 64, chunked prefill, autotuner |
| KV cache | Block reuse, 120 GB host offload, 80% free-GPU-memory fraction |
| Served model name | `openai/gpt-oss-120b` |
| Processors | `harmony` chat processor |
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
    messages=[{"role": "user", "content": "Summarize the rules of chess in three sentences."}],
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

## Validation

Registry PR CI deploys this configuration, calls its playground example, runs the LLM smoke
benchmark, and deactivates the temporary deployment. BIS-LLM must be enabled on the
workspace (Business and Startup workspaces).