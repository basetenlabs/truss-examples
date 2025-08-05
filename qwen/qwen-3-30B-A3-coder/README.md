# Qwen3-Coder-30B-A3B-Instruct Model

This Truss serves the Qwen3-Coder-30B-A3B-Instruct model, a powerful coding-focused language model that excels at agentic coding tasks. The model is based on the [Qwen/Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) model from Hugging Face and is optimized for high-performance coding assistance. It is Apache 2.0 licensed and can be used commercially without restrictions.

## Model Description

The Qwen3-Coder-30B-A3B-Instruct model is a specialized coding language model that features:

- **Agentic Coding**: Excellent performance on agentic coding tasks and browser-use scenarios
- **Long Context**: Native support for 256K tokens, extendable up to 1M tokens with Yarn
- **Function Calling**: Specialized function call format for tool integration
- **Repository-Scale Understanding**: Optimized for understanding large codebases
- **Streaming Support**: Real-time code generation with streaming capabilities

## Model Parameters

The model accepts the following parameters:

- `messages` (required): Array of message objects with role and content
- `model` (optional): Model name (default: "Qwen/Qwen3-Coder-30B-A3B-Instruct")
- `max_tokens` (optional): Maximum tokens to generate (default: 1024)
- `temperature` (optional): Sampling temperature (default: 0.7)
- `stream` (optional): Enable streaming response (default: true)
- `tools` (optional): Array of function definitions for tool calling

## Example Usage

The model outputs structured responses compatible with OpenAI's chat completion format.

```python
import httpx
import os

# Replace with your model ID and API key
model_id = "your-model-id"
baseten_api_key = os.environ["BASETEN_API_KEY"]

# Example 1: Basic code generation
basic_data = {
    "messages": [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Write a quick sort algorithm in Python."}
    ],
    "max_tokens": 1024,
    "temperature": 0.7,
    "stream": True
}

# Example 2: Function calling for tool integration
def square_the_number(num: float) -> dict:
    return {"result": num ** 2}

tools_data = {
    "messages": [
        {"role": "user", "content": "Calculate the square of 1024"}
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "square_the_number",
                "description": "Calculate the square of a number",
                "parameters": {
                    "type": "object",
                    "required": ["num"],
                    "properties": {
                        "num": {
                            "type": "number",
                            "description": "The number to square"
                        }
                    }
                }
            }
        }
    ],
    "max_tokens": 1024,
    "temperature": 0.7
}

# Call the model
print("Generating code...")
response = httpx.post(
    f"https://model-{model_id}.api.baseten.co/development/predict",
    headers={"Authorization": f"Api-Key {baseten_api_key}"},
    json=basic_data,
    timeout=httpx.Timeout(60.0)
)

# Get the result
result = response.json()
print("Generated code:", result.get("choices", [{}])[0].get("message", {}).get("content", ""))
```

## Agentic Coding Examples

The model excels at agentic coding tasks. Here are some example use cases:

```python
# Repository analysis
repo_analysis = {
    "messages": [
        {"role": "user", "content": "Analyze this codebase and suggest improvements for the authentication system."}
    ],
    "max_tokens": 2048
}

# Code review
code_review = {
    "messages": [
        {"role": "user", "content": "Review this Python function for security vulnerabilities:\n\ndef process_user_input(data):\n    return eval(data)"}
    ],
    "max_tokens": 1024
}

# Debugging assistance
debugging = {
    "messages": [
        {"role": "user", "content": "Help me debug this error: 'TypeError: 'NoneType' object is not callable'"}
    ],
    "max_tokens": 1024
}
```

## Best Practices

For optimal performance, we recommend:

1. **Sampling Parameters**:
   - Temperature: 0.7
   - Top-p: 0.8
   - Top-k: 20
   - Repetition penalty: 1.05

2. **Context Length**: Use up to 65,536 tokens for most queries

3. **Streaming**: Enable streaming for real-time code generation

4. **Function Calling**: Define clear tool schemas for agentic tasks

## Deployment

To deploy this model:

1. Clone the repository
2. Make sure you have the Truss CLI installed (`pip install truss`)
3. Run the deployment command:

```bash
truss push qwen/qwen-3-30B-A3-coder --publish
```

## Model Features

- **OpenAI-Compatible API**: Full compatibility with OpenAI's chat completion format
- **Streaming Support**: Real-time response streaming for better user experience
- **Tool Calling**: Native support for function calling and tool integration
- **Long Context**: Handles large codebases and documentation
- **GPU Optimization**: Optimized for H100 GPUs with SGLang

## License

This model is licensed under Apache 2.0 and can be used commercially without restrictions.
