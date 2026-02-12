# Tutorials

Step-by-step guides for learning how to build, configure, and deploy models with Truss. Each tutorial focuses on a single concept and includes a complete, deployable example.

| Directory | Description |
|-----------|-------------|
| [getting-started-bert](getting-started-bert/) | Introductory tutorial packaging a BERT model for deployment |
| [llm-basics](llm-basics/) | Serve a large language model with basic inference |
| [llm-streaming](llm-streaming/) | Stream token-by-token responses from an LLM |
| [image-generation](image-generation/) | Generate images using a diffusion model |
| [speech-to-text](speech-to-text/) | Transcribe audio input to text |
| [cached-weights](cached-weights/) | Use cached model weights for faster cold starts |
| [dynamic-batching](dynamic-batching/) | Batch incoming requests for higher throughput |
| [private-huggingface](private-huggingface/) | Load models from a private HuggingFace repository |
| [system-packages](system-packages/) | Install system-level dependencies in your Truss environment |

## Deploying

Each tutorial can be deployed to Baseten with:

```bash
truss push
```
