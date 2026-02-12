# Nous Capybara 34B Truss (OpenAI Client Compatible)

This is a [Truss](https://truss.baseten.co/) for [Nous Capybara 34B](https://huggingface.co/NousResearch/Nous-Capybara-34B), compatible with our [bridge endpoint for OpenAI ChatCompletion users](https://docs.baseten.co/api-reference/openai).

## Deployment

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd nous-capybara/nous-capybara-34b-openai
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `nous-capybara/nous-capybara-34b-openai` as your working directory, you can deploy the model with:

```sh
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

## Model API reference

This model is designed for our ChatCompletions endpoint:

- [ChatCompletions endpoint tutorial](https://www.baseten.co/blog/gpt-vs-mistral-migrate-to-open-source-llms-with-minor-code-changes/)
- [ChatCompletions endpoint reference docs](https://docs.baseten.co/api-reference/openai)

Note that Nous Capybara currently does not support system messages (see [here](https://huggingface.co/NousResearch/Nous-Capybara-34B/discussions/5)).

See the script below for an example of calling the model, with and without streaming:
```python
from openai import OpenAI
import os

model_id = "YOUR_MODEL_ID"
client = OpenAI(
    api_key="YOUR_BASETEN_API_KEY",
    base_url=f"https://bridge.baseten.co/{model_id}/v1"
)

# Non-streaming example
response = client.chat.completions.create(
    model="nous-capybara-34b",
    messages=[
            {"role": "user", "content": "What happens if I go to the top of the tallest mountain in California with a bucket of water and tip it over the highest cliff?"},
    ],
    stream=False,
)

print(response.choices[0].message.content)

# Streaming example
response = client.chat.completions.create(
    model="nous-capybara-34b",
    messages=[
            {"role": "user", "content": "Who won the world series in 2020?"},
            {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
            {"role": "user", "content": "Where was it played?"}
    ],
    stream=True,
)

for chunk in response:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="")
```
