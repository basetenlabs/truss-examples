# DeepSeek-V4-Flash (BIS-LLM)

This Truss serves DeepSeek's [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
(MIT, the official release superseding the preview) through the Baseten Inference Stack
(TRT-LLM PyTorch backend behind Dynamo cache-aware routing) on four B200 GPUs, with the
checkpoint's built-in DSpark speculative decoding module. It is the registry copy of the
configuration that serves the Baseten library listing `baseten/deepseek-v4-flash`.

The sibling `latency` preset in this directory serves the earlier `deepseek-ai/DeepSeek-V4-Flash`
checkpoint with vLLM and publishes the separate `deepseek-v4-flash-latency` listing.

## Configuration

| Setting | Value |
| --- | --- |
| Checkpoint | `deepseek-ai/DeepSeek-V4-Flash-0731` at `7872f01b`, 48 safetensors shards, about 167 GB |
| Hardware | `B200:4`, tensor parallel 4, MoE expert parallel 4 |
| Engine | BIS-LLM `0.0.1-20260727180920-c08268b2`, TRT-LLM PyTorch backend, `TRTLLM` MoE backend |
| Context | 1,048,576 tokens (`max_seq_len`, `max_input_len`, `tokenizer_limit_length`) |
| Speculative decoding | DSpark, draft length 5, Markov rank 256, target layers 40 to 42; draft weights come from the same checkpoint (`speculative_model_dir` under the mount) |
| KV cache | FP8, 128-token blocks, block reuse, 100 GB host offload, 90% free-GPU-memory fraction |
| Batching | `max_batch_size` 8, `max_num_tokens` 8,192, chunked prefill |
| Served model name | `deepseek-ai/DeepSeek-V4-Flash-0731` |
| Processors | `deepseek_v4` chat processor, reasoning parser, and tool-call parser; arguments as JSON |
| Thinking | Enabled by default; `tokenizer_max_new_tokens_limit` 32,768 |
| Sampling defaults | `top_p` 1, `temperature` 1 |
| Routing | worker tree with residency tracking, overlap score weight 8 |
| Endpoint | OpenAI-compatible `/v1/chat/completions` |

`model_path_for_tokenizer` and `speculative_model_dir` point at `tokenizer/` and `worker/`
under the weights mount. Those folders are not part of the Hugging Face repository; BIS-LLM
derives them as sub-mounts of the single `weights[]` entry (the platform reserves the
`/worker` and `/tokenizer` suffixes for this), as in the deployed listing.

## Usage

```python
from openai import OpenAI

client = OpenAI(
    api_key="<BASETEN_API_KEY>",
    base_url="https://model-<MODEL_ID>.api.baseten.co/environments/production/sync/v1",
)

response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V4-Flash-0731",
    messages=[{"role": "user", "content": "What is the meaning of life?"}],
    max_tokens=32768,
    temperature=1,
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

Tool calling and JSON-schema structured outputs work through the standard OpenAI request
fields.

## Validation

Registry PR CI deploys this configuration, calls its playground example, runs the LLM smoke
benchmark, and deactivates the temporary deployment. BIS-LLM deployments require a
workspace entitled to the referenced `gpuTRTImage`.