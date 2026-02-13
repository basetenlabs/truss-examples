# Custom Server

Examples of deploying models using `docker_server.start_command` instead of a traditional `model.py`. This is useful when deploying models that are already wrapped in an API (e.g. vLLM, SGLang, LMDeploy) or when using a custom Docker image that handles HTTP requests directly.

| Example | Engine | Description |
|---------|--------|-------------|
| [llama3-8b-instruct-sglang](llama3-8b-instruct-sglang/) | SGLang | Llama 3 8B Instruct via SGLang server |
| [llama3-70b-instruct-sglang](llama3-70b-instruct-sglang/) | SGLang | Llama 3 70B Instruct via SGLang server |
| [llama3-70b-instruct-lmdeploy](llama3-70b-instruct-lmdeploy/) | LMDeploy | Llama 3 70B Instruct via LMDeploy server |
| [llama3-8b-instruct-lmdeploy](llama3-8b-instruct-lmdeploy/) | LMDeploy | Llama 3 8B Instruct via LMDeploy server |
| [deepseek-v2-5-instruct-sglang](deepseek-v2-5-instruct-sglang/) | SGLang | DeepSeek v2.5 Instruct via SGLang server |
| [pixtral-12b](pixtral-12b/) | vLLM | Pixtral 12B multimodal model |
| [infinity-embedding-server](infinity-embedding-server/) | Infinity | Embedding server using Infinity engine |
| [ultravox-0.4](ultravox-0.4/) | vLLM | Ultravox 0.4 multimodal audio model |
| [ultravox-0.5-8b](ultravox-0.5-8b/) | vLLM | Ultravox 0.5 8B multimodal audio model |
| [ultravox-0.6-70b](ultravox-0.6-70b/) | vLLM | Ultravox 0.6 70B multimodal audio model |
| [voxtral-mini-3b-2507](voxtral-mini-3b-2507/) | vLLM | Voxtral Mini 3B speech model |
| [voxtral-small-24b-2507](voxtral-small-24b-2507/) | vLLM | Voxtral Small 24B speech model |

## Deploy

```sh
truss push infrastructure/custom-server/llama3-8b-instruct-sglang
```

## How it works

Instead of writing a `model.py` with `load()` and `predict()` methods, these examples use `docker_server.start_command` in `config.yaml` to launch an existing inference server:

```yaml
docker_server:
  start_command: "python -m sglang.launch_server --model meta-llama/Meta-Llama-3-8B-Instruct ..."
  predict_endpoint: /v1/chat/completions
  server_port: 8000
```

This avoids unnecessary overhead when the model already provides its own HTTP endpoint.
