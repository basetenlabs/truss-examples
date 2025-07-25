# vLLM Truss to deploy chat completion model

## Two options

### Vllm server via vllm serve (Recommended)
This is an openai-compatible codeless solution that the large majority of users should use when deploying with vLLM. The only work required to set up this vLLM server is to modify the configs in the vllm serve command in `config.yaml`. See `vllm_server` directory for the truss.

### Vllm using truss server
This solution is for a custom deployment where you require flexibility such as custom inference logic. This is OpenAI compatible with a few modifications in your calling code. See `truss_server` directory for the truss.