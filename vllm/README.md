# vLLM Truss to deploy chat completion model

## What is this Truss example doing

This is a general purpose [Truss](https://truss.baseten.co/) that can deploy an asynchronous vLLM engine([AsyncLLMEngine](https://docs.vllm.ai/en/latest/dev/engine/async_llm_engine.html#asyncllmengine)) of any customized configuration with [all compatible chat completion models](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Two options

### Vllm server via vllm serve (Recommended)
This is an openai-compatible codeless solution that the large majority of users should use when deploying with vLLM. The only work required to set up this vLLM server is to modify the configs in the vllm serve command in `config.yaml`.

### Vllm using truss server
This solution is for a more custom deployment where you require flexibility such as custom logic in your predictions.