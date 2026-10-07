# Ling 3.0 Flash INT4 — latency preset

Ling 3.0 Flash is InclusionAI's hybrid-linear sparse mixture-of-experts reasoning
model. It has 124B total parameters, activates 5.1B parameters per token, and
alternates 35 Kimi Delta Attention layers with 7 gated MLA layers. This preset
serves the groupwise INT4 checkpoint on two H100 GPUs through vLLM's
OpenAI-compatible `/v1/chat/completions` endpoint.

## Serving contract

- **Served model name:** `inclusionAI/Ling-3.0-flash-int4`
- **Hardware:** `H100:2`, tensor parallel size 2
- **Context:** 262,144 tokens (the model's 256K trained context stage)
- **Reasoning:** parsed with vLLM's `ling3` reasoning parser
- **Tool calling:** automatic tool choice with vLLM's `ling3` tool parser
- **Recommended sampling:** `temperature=0.6`, `top_p=0.95`, `top_k=20`
- **Thinking:** enabled by default by the model; callers can explicitly pass
  `chat_template_kwargs: {"enable_thinking": true}`

The checkpoint uses compressed-tensors groupwise INT4 weights for linear layers;
unquantized operations use BF16. Prefix caching, chunked prefill, and full plus
piecewise CUDA graphs are enabled.

## Serving stack choice

The model card currently recommends InclusionAI's Ling-specific SGLang branch,
while the requested deployment target is vLLM. Upstream vLLM now contains Ling 3
model loading plus the `ling3` tool and reasoning parsers, so this preset uses that
path and leaves runtime confirmation to registry PR CI. BIS-LLM was not selected:
although Ling is a large MoE, BIS-LLM requires workspace-specific Enterprise image
settings and cannot participate in the registry's development-deployment test loop.

## Reproducibility

- **Weights:** `inclusionAI/Ling-3.0-flash-int4` at immutable revision
  `959d3a48cf05d2daf5fe7bdbbd3a6bb119e359f2`
- **License:** MIT
- **vLLM:** commit `5a4c8d99242e9e069b604d0e9b969e77f7dd501d`, pinned by
  both Docker tag and manifest digest

Ling 3 model loading and its `ling3` tool/reasoning parsers landed in vLLM after
the v0.27.1 release, so this preset intentionally uses a pinned nightly image.
Replace it with the first stable vLLM release containing that support once one is
available.

## Example request

```json
{
  "model": "inclusionAI/Ling-3.0-flash-int4",
  "messages": [
    {
      "role": "user",
      "content": "Explain why the sky appears blue in three concise sentences."
    }
  ],
  "stream": true,
  "max_tokens": 512,
  "temperature": 0.6,
  "top_p": 0.95,
  "top_k": 20,
  "chat_template_kwargs": {
    "enable_thinking": true
  }
}
```

This is a starting configuration. Registry PR CI should be used to validate model
startup, the 256K memory profile, chat streaming, reasoning output, and tool calls
before the preset is considered production-ready.