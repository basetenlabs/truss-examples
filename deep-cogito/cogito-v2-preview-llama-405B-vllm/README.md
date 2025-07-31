# Cogito v2 Llama 405B Truss (vLLM)

Cogito's Llama-based 405B model has powerful tool calling and reasoning capabilities. See this [blog post](https://www.deepcogito.com/research/cogito-v2-preview).

This is a [Truss](https://truss.baseten.co/) to deploy the model using the vLLM OpenAI Compatible server. This model requires 8x B200 GPUs to deploy. Users should contact [support@baseten.co](mailto:support@baseten.co) before deploying.

## Deployment

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples.git
cd deep-cogito/cogito-v2-preview-llama-405B-vllm
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`
3. Retrieve your Hugging Face token from the [settings](https://huggingface.co/settings/tokens).
4. Set your Hugging Face token as a Baseten secret [here](https://app.baseten.co/settings/secrets) with the key `hf_access_token`. Note that you will *not* be able to successfully deploy the model without doing this.

With `cogito-v2-preview-llama-405B-vllm` as your working directory, you can deploy the model with:

```sh
truss push --publish
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

## vLLM OpenAI Compatible Server

This Truss demonstrates how to start [vLLM's OpenAI compatible server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) without the need for a `model.py` through the `docker_server.start_command` option.

## API Documentation

The API follows the OpenAI ChatCompletion format. You can interact with the model using the standard ChatCompletion interface.

Example usage:

```python
from openai import OpenAI

model_id = "your-model-id" # Replace with your model ID

client = OpenAI(
    api_key="YOUR-API-KEY",
    base_url=f"https://model-{model_id}.api.baseten.co/environments/production/sync/v1"
)

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
                "required": [
                    "location"
                ]
            }
        }
    }
]

# Example usage of the OpenAI client to use a tool call
response = client.chat.completions.create(
    model="llama",
    messages=[
        {
            "role": "user",
            "content": "What is today's temperature in celsius? I'm in Paris."
        }
    ],
    tools=tools,
)

print(response.json())
```

## Model Details

- **Model**: Cogito v2 Preview Llama 405B
- **Architecture**: Dense transformer
- **GPU Requirements**: 8x B200
- **Tool Call Parser**: llama3_json
- **Features**: Prefix caching, chunked prefill
- **Tensor Parallel Size**: 8

## Support

If you have any questions or need assistance, please open an issue in this repository or contact our [support team](mailto:support@baseten.co).
