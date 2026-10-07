# Qwen3.6 35B-A3B NVFP4 — throughput preset

This Truss serves [RedHatAI/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/RedHatAI/Qwen3.6-35B-A3B-NVFP4)
through vLLM on a single B200 GPU. Qwen3.6-35B-A3B is a mixture-of-experts model with
35B total / 3B active parameters, quantized to NVFP4 (4-bit float) with FP8 de-scales.
It supports a 262,144-token context window, reasoning, tool calling, and a built-in
multi-token-prediction draft head.

## Configuration

| Setting | Value |
| --- | --- |
| Checkpoint | `RedHatAI/Qwen3.6-35B-A3B-NVFP4@main` (gated; requires `hf_access_token`) |
| Precision | NVFP4 weights; FP8 MoE de-scales |
| Hardware | `B200` |
| Context | 262,144 tokens |
| Maximum concurrent sequences | 512 |
| Server | `vllm/vllm-openai:v0.28.0` |
| Endpoint | OpenAI-compatible `/v1/chat/completions` |
| Served model name | `RedHatAI/Qwen3.6-35B-A3B-NVFP4` |
| MoE backend | `flashinfer_cutlass` (FlashInfer, FP4 + FP8) |
| Speculative decoding | `qwen3_5_mtp`, 3 speculative tokens |
| Tool parser | `qwen3_coder` |
| Reasoning parser | `qwen3` |
| Weight loading | `runai_streamer` (BDN streaming, no disk materialization) |

## Usage

```python
from openai import OpenAI

client = OpenAI(
    api_key="<BASETEN_API_KEY>",
    base_url="https://model-<MODEL_ID>.api.baseten.co/environments/production/sync/v1",
)

response = client.chat.completions.create(
    model="RedHatAI/Qwen3.6-35B-A3B-NVFP4",
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ],
    max_tokens=100,
    temperature=0.7,
)

print(response.choices[0].message.content)
```

## Sources and validation

- [RedHatAI/Qwen3.6-35B-A3B-NVFP4 model card](https://huggingface.co/RedHatAI/Qwen3.6-35B-A3B-NVFP4)

The model-registry PR deployment and smoke benchmark validate this B200
configuration before merge.