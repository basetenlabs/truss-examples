# Ornith 1.5 35B-A3B

This preset serves the MIT-licensed
[`ornith-ai/Ornith-1.5-35B-A3B-FP8`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-FP8)
checkpoint with vLLM on two 80 GB H100 GPUs. Ornith 1.5 is a multimodal
mixture-of-experts reasoning model for coding and agentic tasks, with about
35.95B total parameters and about 3B activated parameters per token.

## Serving contract

- **Endpoint:** OpenAI-compatible `POST /v1/chat/completions`
- **Served model name:** `Ornith-1.5-35B-A3B`
- **Hardware:** `H100:2` with tensor parallel size 2
- **Context limit:** 262,144 combined input and output tokens
- **Modalities:** text and image input, with at most one image per request
- **Reasoning:** returned separately in `reasoning_content` by vLLM's `qwen3` parser
- **Tool calls:** returned as OpenAI-style `tool_calls` by vLLM's `qwen3_xml` parser

The configuration follows the model card's vLLM recipe and pins the minimum
supported stable image, `vllm/vllm-openai:v0.19.1`. It enables prefix caching,
uses 90% GPU-memory utilization, and preserves the checkpoint's native 256K
context rather than enabling the optional YaRN extension.

Recommended sampling for general tasks is `temperature: 0.6`, `top_p: 0.95`,
and `top_k: 20`. The model reasons by default; its chat template opens assistant
turns with a `<think>` block, which the configured reasoning parser separates
from the final answer.

## Example request

```json
{
  "model": "Ornith-1.5-35B-A3B",
  "messages": [
    {
      "role": "user",
      "content": "Write a one-line Python lambda that squares a number."
    }
  ],
  "temperature": 0.6,
  "top_p": 0.95,
  "max_tokens": 1024
}
```

The checkpoint is ungated. The registry config still declares the shared
`hf_access_token` secret so CI can mirror the pinned Hugging Face revision to
the Baseten Delivery Network.