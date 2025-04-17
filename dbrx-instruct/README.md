# DBRX-Instruct Truss

DBRX-Instruct is a state-of-the-art language model developed by Anthropic, leveraging the latest advancements in constitutional AI to ensure safe and effective instruction-following capabilities. This model is particularly adept at understanding and generating human-like text, making it an invaluable tool for a wide range of applications including content creation, summarization, and question-answering tasks.

## Deploying DBRX-Instruct Truss

To deploy the DBRX-Instruct Truss on Baseten:

1. Clone this repo: `git clone https://github.com/baseten/truss-examples`

2. Make sure you have a [Baseten account](https://app.baseten.co/signup).

3. Install Truss: `npm install -g @baseten/truss`

4. Log in to your Baseten account using an [API key](https://docs.baseten.co/api_keys/).

5. Deploy: `truss deploy dbrx-instruct` (you may be prompted to redeploy the model if you've deployed previously).

## Hardware

To deploy DBRX-Instruct you'll need the following resources in the Baseten cloud:

- 2 CPUs
- 32GB RAM
- 1 A100 GPU Accelerator

If your account does not have access to A100 GPUs, you can modify the `config.yaml` to use a different accelerator.

## API

The DBRX-Instruct Truss has a single predict route that accepts a JSON payload with the following parameters:

| Parameter     | Type              | Description                                                                                                             |
|---------------|-------------------|-------------------------------------------------------------------------------------------------------------------------|
| `prompt`      | string (required) | The prompt to use for generating text.                                                                                  |
| `max_tokens`  | integer           | The maximum number of tokens to generate in the output. Defaults to 512.                                                |
| `temperature` | float             | Controls the "creativity" of the generated text. Higher values (e.g. 1.0) produce more diverse outputs. Defaults to 0.5. |

Example payload:
```json
{
  "prompt": "Write a haiku about constitutional AI.",
  "max_tokens": 128,
  "temperature": 0.8
}
```

## Example Usage

You can invoke the model via a REST API:

```bash
curl -X POST https://your-app-url.baseten.co/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Write a haiku about constitutional AI.",
    "max_tokens": 128,
    "temperature": 0.8
  }'
```

Or using the Baseten Python client:

```python
import baseten

# Get the deployed model
model = baseten.deployed_model_id('your-deployed-model-id')

# Get the model's predict route
predict = model.predict

# Make a prediction
response = predict(
  prompt="Write a haiku about constitutional AI.",
  max_tokens=128,
  temperature=0.8
)
print(response)
```

## Generation Parameters and Limitations

The DBRX-Instruct model allows for customization of the generation process through parameters such as `max_tokens` and `temperature`. These parameters enable users to control the length and creativity of the generated text. However, it's important to note that increasing `max_tokens` significantly can impact the response time and computational resources required. Similarly, a higher `temperature` can lead to more varied and creative outputs but may also increase the risk of generating off-topic or nonsensical text.

## Optimal Use Cases

DBRX-Instruct excels in scenarios requiring nuanced understanding and generation of text, such as:
- Generating high-quality, contextually relevant content for articles or blogs.
- Summarizing long documents or articles into concise paragraphs.
- Answering questions based on provided context or knowledge.

For best results, it's recommended to provide clear, context-rich prompts and to experiment with generation parameters to find the optimal settings for your specific use case.
