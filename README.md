# Truss Examples

[![Truss Examples CI](https://github.com/basetenlabs/truss-examples/actions/workflows/test-examples.yml/badge.svg)](https://github.com/basetenlabs/truss-examples/actions/workflows/test-examples.yml)

Production-ready **inference** examples for [Truss](https://truss.baseten.co/),
the simplest way to serve AI/ML models. Each example is ready to deploy as-is or
adapt to your own use case.

> **Looking for training and fine-tuning?** See
> [ml-cookbook](https://github.com/basetenlabs/ml-cookbook) for training
> recipes, LoRA fine-tuning workflows, and end-to-end training examples on
> Baseten.

## Quick start

Clone the repository:

```bash
git clone https://github.com/basetenlabs/truss-examples
cd truss-examples
```

Install Truss:

```bash
pip install --upgrade truss
```

Deploy any example:

```bash
truss push tutorials/getting-started-bert
```

You will be prompted for an API key. Get one from the [Baseten API keys page](https://app.baseten.co/settings/account/api_keys).

See individual example READMEs for invocation details specific to each model.

## Repository structure

| Category                          | Description                                                                                                                                       | Examples             | Path              |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- | ----------------- |
| [Tutorials](tutorials/)           | Getting-started guides covering core Truss features: BERT, LLMs, streaming, image generation, speech-to-text, caching, dynamic batching, and more | 9                    | `tutorials/`      |
| [LLMs](llm/)                      | Large language model families including Llama, Qwen, DeepSeek, Mistral, Gemma, Phi, Falcon, and others                                            | 19 families          | `llm/`            |
| [Embeddings](embeddings/)         | Embedding and reranking models via BEI (48 models), TEI (15 models), CLIP, and text-embeddings-inference                                          | 4 engines, 65 models | `embeddings/`     |
| [Image](image/)                   | Image generation, editing, and segmentation: Stable Diffusion, Flux, SDXL, ComfyUI, ControlNet, SAM, and more                                     | 15                   | `image/`          |
| [Audio](audio/)                   | Speech-to-text, text-to-speech, and audio generation: Whisper, Kokoro, Chatterbox, Orpheus, XTTS, MusicGen, and more                              | 19                   | `audio/`          |
| [Optimized](optimized/)           | Production-grade optimized configs via Briton (33 TRT-LLM models) and BISv2 (11 models)                                                           | 2 engines, 44 models | `optimized/`      |
| [Infrastructure](infrastructure/) | Patterns and techniques: custom servers, gRPC, Chains, multiprocessing, metrics, model caching, and more                                          | 14                   | `infrastructure/` |

### Tutorials

Step-by-step introductions to Truss concepts and features.

| Example                                                 | Description                                 |
| ------------------------------------------------------- | ------------------------------------------- |
| [getting-started-bert](tutorials/getting-started-bert/) | Deploy a BERT model for text classification |
| [llm-basics](tutorials/llm-basics/)                     | Serve an LLM with Truss                     |
| [llm-streaming](tutorials/llm-streaming/)               | Stream LLM responses token-by-token         |
| [image-generation](tutorials/image-generation/)         | Serve an image generation model             |
| [speech-to-text](tutorials/speech-to-text/)             | Deploy a speech-to-text pipeline            |
| [cached-weights](tutorials/cached-weights/)             | Cache model weights for faster cold starts  |
| [private-huggingface](tutorials/private-huggingface/)   | Access private Hugging Face models          |
| [dynamic-batching](tutorials/dynamic-batching/)         | Enable dynamic batching for throughput      |
| [system-packages](tutorials/system-packages/)           | Add system-level dependencies               |

### LLMs

Deploy popular large language model families with optimized serving configurations.

| Model family      | Path                                               |
| ----------------- | -------------------------------------------------- |
| Llama             | [`llm/llama/`](llm/llama/)                         |
| Qwen              | [`llm/qwen/`](llm/qwen/)                           |
| DeepSeek          | [`llm/deepseek/`](llm/deepseek/)                   |
| Mistral           | [`llm/mistral/`](llm/mistral/)                     |
| Gemma             | [`llm/gemma/`](llm/gemma/)                         |
| Phi               | [`llm/phi/`](llm/phi/)                             |
| Falcon            | [`llm/falcon/`](llm/falcon/)                       |
| Cogito            | [`llm/cogito/`](llm/cogito/)                       |
| CogVLM            | [`llm/cogvlm/`](llm/cogvlm/)                       |
| LLaVA             | [`llm/llava/`](llm/llava/)                         |
| LoRA              | [`llm/lora/`](llm/lora/)                           |
| OpenAI-compatible | [`llm/openai/`](llm/openai/)                       |
| Midnight          | [`llm/midnight/`](llm/midnight/)                   |
| MiniMax           | [`llm/minimax/`](llm/minimax/)                     |
| Nemotron          | [`llm/nemotron/`](llm/nemotron/)                   |
| NSQL              | [`llm/nsql/`](llm/nsql/)                           |
| Personaplex       | [`llm/personaplex-7b-v1/`](llm/personaplex-7b-v1/) |
| Seed              | [`llm/seed/`](llm/seed/)                           |
| Z-AI              | [`llm/z-ai/`](llm/z-ai/)                           |

### Embeddings

Embedding, reranking, and classification models across multiple serving engines.

| Engine                    | Models | Path                                                                             |
| ------------------------- | ------ | -------------------------------------------------------------------------------- |
| BEI                       | 48     | [`embeddings/bei/`](embeddings/bei/)                                             |
| TEI                       | 15     | [`embeddings/tei/`](embeddings/tei/)                                             |
| CLIP                      | 1      | [`embeddings/clip/`](embeddings/clip/)                                           |
| text-embeddings-inference | 1      | [`embeddings/text-embeddings-inference/`](embeddings/text-embeddings-inference/) |

### Image

Image generation, editing, upscaling, and segmentation models.

| Example             | Path                                                               |
| ------------------- | ------------------------------------------------------------------ |
| Stable Diffusion    | [`image/stable-diffusion/`](image/stable-diffusion/)               |
| Flux                | [`image/flux/`](image/flux/)                                       |
| Flux Dev TRT (B200) | [`image/flux-dev-trt-b200/`](image/flux-dev-trt-b200/)             |
| Sana                | [`image/sana/`](image/sana/)                                       |
| ComfyUI             | [`image/comfyui/`](image/comfyui/)                                 |
| ControlNet QR Code  | [`image/control-net-qrcode/`](image/control-net-qrcode/)           |
| DeepFloyd XL        | [`image/deepfloyd-xl/`](image/deepfloyd-xl/)                       |
| Fotographer         | [`image/fotographer/`](image/fotographer/)                         |
| GFP-GAN             | [`image/gfp-gan/`](image/gfp-gan/)                                 |
| IP-Adapter          | [`image/ip-adapter/`](image/ip-adapter/)                           |
| Magic Animate       | [`image/magic-animate/`](image/magic-animate/)                     |
| Playground v2       | [`image/playground-v2-aesthetic/`](image/playground-v2-aesthetic/) |
| Segment Anything    | [`image/segment-anything/`](image/segment-anything/)               |
| DIS Segmentation    | [`image/dis-segmentation/`](image/dis-segmentation/)               |
| Image Segmentation  | [`image/image-segmentation/`](image/image-segmentation/)           |

### Audio

Speech-to-text, text-to-speech, and audio/music generation models.

| Example                    | Path                                                                 |
| -------------------------- | -------------------------------------------------------------------- |
| Whisper                    | [`audio/whisper/`](audio/whisper/)                                   |
| Kokoro                     | [`audio/kokoro/`](audio/kokoro/)                                     |
| Chatterbox TTS             | [`audio/chatterbox-tts/`](audio/chatterbox-tts/)                     |
| Piper TTS                  | [`audio/piper-tts/`](audio/piper-tts/)                               |
| XTTS v2                    | [`audio/xtts-v2/`](audio/xtts-v2/)                                   |
| XTTS Streaming             | [`audio/xtts-streaming/`](audio/xtts-streaming/)                     |
| Orpheus 3B (WebSockets)    | [`audio/orpheus-3b-websockets/`](audio/orpheus-3b-websockets/)       |
| Orpheus (Best Performance) | [`audio/orpheus-best-performance/`](audio/orpheus-best-performance/) |
| Sesame CSM 1B              | [`audio/sesame-csm-1b/`](audio/sesame-csm-1b/)                       |
| MetaVoice 1B               | [`audio/metavoice-1b/`](audio/metavoice-1b/)                         |
| Ultravox                   | [`audio/ultravox/`](audio/ultravox/)                                 |
| AudioGen Medium            | [`audio/audiogen-medium/`](audio/audiogen-medium/)                   |
| MusicGen Large             | [`audio/musicgen-large/`](audio/musicgen-large/)                     |
| MusicGen Melody            | [`audio/musicgen-melody/`](audio/musicgen-melody/)                   |
| NVIDIA Parakeet            | [`audio/nvidia-parakeet/`](audio/nvidia-parakeet/)                   |
| Qwen ASR                   | [`audio/qwen-asr/`](audio/qwen-asr/)                                 |
| Qwen Omni                  | [`audio/qwen-omni/`](audio/qwen-omni/)                               |
| Qwen Omni Thinker          | [`audio/qwen-omni-thinker/`](audio/qwen-omni-thinker/)               |
| Voxtral Streaming 4B       | [`audio/voxtral-streaming-4b/`](audio/voxtral-streaming-4b/)         |

### Optimized

Production-grade, autogenerated model configurations using TRT-LLM and other optimization engines.

| Engine           | Models | Path                                     |
| ---------------- | ------ | ---------------------------------------- |
| Briton (TRT-LLM) | 33     | [`optimized/briton/`](optimized/briton/) |
| BISv2            | 11     | [`optimized/bisv2/`](optimized/bisv2/)   |

### Infrastructure

Patterns, techniques, and advanced serving configurations.

| Example                       | Path                                                                                             |
| ----------------------------- | ------------------------------------------------------------------------------------------------ |
| Custom Server                 | [`infrastructure/custom-server/`](infrastructure/custom-server/)                                 |
| gRPC                          | [`infrastructure/grpc/`](infrastructure/grpc/)                                                   |
| Chains                        | [`infrastructure/chains-examples/`](infrastructure/chains-examples/)                             |
| Multiprocessing               | [`infrastructure/multiprocessing/`](infrastructure/multiprocessing/)                             |
| Metrics                       | [`infrastructure/metrics/`](infrastructure/metrics/)                                             |
| Model Cache                   | [`infrastructure/model-cache/`](infrastructure/model-cache/)                                     |
| Custom Engine Builder Control | [`infrastructure/custom-engine-builder-control/`](infrastructure/custom-engine-builder-control/) |
| LLama.cpp Server              | [`infrastructure/llama-cpp-server/`](infrastructure/llama-cpp-server/)                           |
| JSON Formatter                | [`infrastructure/jsonformatter/`](infrastructure/jsonformatter/)                                 |
| LayoutLM Document QA          | [`infrastructure/layoutlm-document-qa/`](infrastructure/layoutlm-document-qa/)                   |
| N-gram Speculator             | [`infrastructure/ngram-speculator/`](infrastructure/ngram-speculator/)                           |
| PaddlePaddle                  | [`infrastructure/paddlepaddle/`](infrastructure/paddlepaddle/)                                   |
| Autodesk WALA                 | [`infrastructure/autodesk-wala/`](infrastructure/autodesk-wala/)                                 |
| Binocular                     | [`infrastructure/binocular/`](infrastructure/binocular/)                                         |

## Contributing

We welcome new models and improvements to existing examples. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.
