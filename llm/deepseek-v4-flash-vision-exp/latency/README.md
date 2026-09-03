# DeepSeek V4 Flash Vision (Experimental) — latency preset

DeepSeek-V4-Flash-Vision-Exp is DeepSeek's experimental multimodal variant of
DeepSeek V4 Flash: the same 305B-total-parameter FP8 MoE base (256 routed
experts, 6 active per token, DFlash attention, Hyper-Connections, DSpark
forward path) with an added vision encoder and aligner for image-text-to-text
agent tasks. This preset serves the FP8 checkpoint on four B200 GPUs through
vLLM's OpenAI-compatible `/v1/chat/completions` endpoint.

## Serving contract

- **Served model name:** `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`
- **Hardware:** pinned `instance_type: B200:4`, tensor parallel size 4 with
  expert parallelism
- **Modalities:** text + images (up to 2 images per prompt via
  `--limit-mm-per-prompt.image 2`; OpenAI-style `image_url` content blocks)
- **Context:** `--max-model-len auto` (native 64K, YaRN-extended to 1M
  positions), matching the `deepseek-v4-flash` preset
- **Speculative decoding:** native DSpark (semi-autoregressive parallel
  drafting) with 3 speculative tokens — one per shipped draft layer
  (`num_nextn_predict_layers: 3`; vLLM requires a multiple of it). The draft
  layers live inside the main weight shards and include a confidence head,
  which the plain `mtp` loader cannot load — DSpark is the only spec-decode
  method compatible with this checkpoint
- **Reasoning:** parsed with vLLM's `deepseek_v4` reasoning parser
- **Tool calling:** automatic tool choice with vLLM's `deepseek_v4` tool parser
- **Recommended sampling:** `temperature=1.0`, `top_p=0.95` (model card)

## Differences from the `deepseek-v4-flash` preset

- **Vision flags.** `--limit-mm-per-prompt.image 2` (dotted-flag form) enables
  image input; the vision encoder is small relative to the base model
  (32 layers, dim 1024, ≤384 tokens per image).
- **Static tensor parallelism.** A plain `vllm serve` command with
  `--tensor-parallel-size 4` instead of the `sh -c 'GPU_COUNT=...'` wrapper.

## Reproducibility

- **Weights:** `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` at immutable revision
  `e46e16bf6035c6f317eb2ac7458eb0362926d402`
- **License:** MIT (ungated; the `hf_access_token` secret is declared per
  registry convention)
- **Serving image:** `vllm/vllm-openai` at immutable nightly commit
  `27a94d1c`, pinned by both tag and manifest digest — the first nightly
  containing DeepSeek-V4-Flash-Vision-Exp support
  ([vllm#54566](https://github.com/vllm-project/vllm/pull/54566), merged
  2026-09-02). Replace with the first stable vLLM release containing that
  support once one is available

## Example request

```json
{
  "model": "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
  "messages": [
    {
      "role": "user",
      "content": [
        { "type": "text", "text": "Describe this image in one sentence." },
        {
          "type": "image_url",
          "image_url": { "url": "https://picsum.photos/id/237/200/300" }
        }
      ]
    }
  ],
  "stream": true,
  "max_tokens": 32768,
  "temperature": 1.0,
  "top_p": 0.95
}
```

This is a starting configuration. Registry PR CI should be used to validate
model startup, the multimodal path, chat streaming, reasoning output, and tool
calls before the preset is considered production-ready.