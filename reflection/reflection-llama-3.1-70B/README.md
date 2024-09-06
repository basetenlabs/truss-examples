# Reflection Llama 3.1 70B

This model is fine-tuned from Llama 3.1 for better reasoning and benchmark performance. See [Hugging Face model card](https://huggingface.co/mattshumer/Reflection-Llama-3.1-70B) for details.

To push, run

```
truss push --publish
```

For inference, make sure to use the exact system prompt. e.g.

```python
import requests
import os

model_id = "" # Paste your model ID from your Baseten dashboard
baseten_api_key = os.environ["BASETEN_API_KEY"]

# Call model endpoint
resp = requests.post(
    f"https://model-{model_id}.api.baseten.co/production/predict",
    headers={"Authorization": f"Api-Key {baseten_api_key}"},
    json={
      "messages": [
          {"role": "system", "content": "You are a world-class AI system, capable of complex reasoning and reflection. Reason through the query inside <thinking> tags, and then provide your final response inside <output> tags. If you detect that you made a mistake in your reasoning at any point, correct yourself inside <reflection> tags."},
          {"role": "user", "content": "Could Albert Einstein have visited the Ottoman Empire?"}
      ],
      "max_tokens": 1024,
      "temperature": 0.7,
      "top_p": 0.95
    },
    stream=True
)

# Print the generated tokens as they get streamed
for content in resp.iter_content():
    print(content.decode("utf-8"), end="", flush=True)
```
