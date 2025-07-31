# Cogito v2 Preview Models

This directory contains [Truss](https://truss.baseten.co/) configurations for deploying Cogito v2 preview models using vLLM's OpenAI-compatible server. These models feature powerful tool calling and reasoning capabilities, as detailed in the [Cogito v2 research blog post](https://www.deepcogito.com/research/cogito-v2-preview).

## Available Models

| Model | Size | Architecture | GPU Requirements |
|-------|------|--------------|------------------|
| [Cogito v2 Llama 70B](./cogito-v2-preview-llama-70B-vllm/) | 70B | Dense | 2x H100 |
| [Cogito v2 Llama 109B MoE](./cogito-v2-preview-llama-109B-MoE-vllm/) | 109B | MoE (Mixture of Experts) | 4x H100 |
| [Cogito v2 Llama 405B](./cogito-v2-preview-llama-405B-vllm/) | 405B | Dense | 8x B200 |
| [Cogito v2 DeepSeek 671B MoE](./cogito-v2-preview-deepseek-671B-MoE-vllm/) | 671B | MoE (Mixture of Experts) | 8x B200 |




## Prerequisites

Before deploying any of these models, ensure you have:

1. **Baseten Account**: Sign up at [app.baseten.co/signup](https://app.baseten.co/signup)
2. **API Key**: Get your API key from [app.baseten.co/settings/api_keys](https://app.baseten.co/settings/api_keys)
3. **Truss Installation**: Install the latest version: `pip install --upgrade truss`
4. **Hugging Face Token**: Retrieve from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
5. **Baseten Secret**: Set your HF token as a Baseten secret with key `hf_access_token` at [app.baseten.co/settings/secrets](https://app.baseten.co/settings/secrets)

## Deployment

### Quick Start

1. Clone this repository:
```bash
git clone https://github.com/basetenlabs/truss-examples.git
cd cogito
```

2. Navigate to your desired model directory:
```bash
cd cogito-v2-preview-llama-70B-vllm  # or any other model
```

3. Deploy the model:
```bash
truss push --publish
```

### GPU Requirements

- **B200 GPUs**: Required for 405B and 671B models (contact [support@baseten.co](mailto:support@baseten.co) before deploying)
- **H100 GPUs**: Required for 70B and 109B MoE models

## API Usage

All models follow the OpenAI ChatCompletion format. Here's an example using the Python client for tool calling:

```python
from openai import OpenAI

# Replace with your model ID after deployment
model_id = "your-model-id"  # e.g. "yqvy46gq"
client = OpenAI(
    api_key="YOUR-API-KEY",
    base_url=f"https://model-{model_id}.api.baseten.co/environments/production/sync/v1"
)

# Example tool calling
def get_temperature_in_celsius(location=None):
    return 22

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_temperature_in_celsius",
            "description": "Get the current temperature in celsius.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get the temperature for."
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# Chat completion with tool calling
response = client.chat.completions.create(
    model="llama",  # or "deepseek" for DeepSeek model
    messages=[
        {
            "role": "user",
            "content": "What is today's temperature in celsius? I'm in Paris."
        }
    ],
    tools=tools,
    stream=True,
    max_tokens=1000,
    temperature=0.6
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```


## Support

- **Documentation**: [Truss documentation](https://truss.baseten.co)
- **Issues**: Open an issue in this repository
- **Support**: Contact [support@baseten.co](mailto:support@baseten.co)
- **Research**: [Cogito v2 research blog](https://www.deepcogito.com/research/cogito-v2-preview)

## License

These models are subject to the respective licenses of the underlying model weights. Please refer to the Hugging Face model pages for specific licensing information.
