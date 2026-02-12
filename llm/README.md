# Large Language Models

Production-ready Truss configurations for large language models spanning a wide range of model families, sizes, and serving engines (vLLM, SGLang, TRT-LLM). Many directories contain multiple variants optimized for different hardware or quantization levels.

| Directory | Models | Description |
|-----------|--------|-------------|
| [llama](llama/) | 14 | Meta Llama 3.x and 4.x models including 8B, 70B, 405B, vision, and TRT-LLM engine builds |
| [qwen](qwen/) | 28 | Alibaba Qwen 2.5 and Qwen 3 models including coder, math, vision, and MoE variants |
| [mistral](mistral/) | 15 | Mistral and Mixtral models with vLLM, TRT-LLM, and Devstral engine builds |
| [deepseek](deepseek/) | 7 | DeepSeek R1 distilled models and vision/OCR variants |
| [nemotron](nemotron/) | 7 | NVIDIA Nemotron models including Nano, Ultra, and vision variants |
| [z-ai](z-ai/) | 5 | Zhipu GLM-4 models in various sizes and quantizations |
| [cogito](cogito/) | 4 | Deep Cogito v2 Preview models on Llama and DeepSeek backbones |
| [gemma](gemma/) | 3 | Google Gemma 2 and 3 models served with vLLM |
| [phi](phi/) | 3 | Microsoft Phi-3 and Phi-3.5 mini instruction-tuned models |
| [llava](llava/) | 3 | LLaVA multimodal vision-language models (v1.5, v1.6) |
| [lora](lora/) | 3 | LoRA adapter serving with vLLM, SGLang, and TRT-LLM engines |
| [falcon](falcon/) | 1 | TII Falcon 3 model with TRT-LLM engine |
| [openai](openai/) | 2 | GPT-OSS 20B and 120B open-source reproductions |
| [minimax](minimax/) | 1 | MiniMax M2-1 model |
| [cogvlm](cogvlm/) | 1 | CogVLM visual question answering model |
| [midnight](midnight/) | 1 | Midnight model for text generation |
| [nsql](nsql/) | 1 | NSQL natural language to SQL model |
| [personaplex-7b-v1](personaplex-7b-v1/) | 1 | PersonaPlex 7B persona-driven chat model |
| [seed](seed/) | 1 | Seed LLM model |

## Deploying

Each model can be deployed to Baseten with:

```bash
truss push
```
