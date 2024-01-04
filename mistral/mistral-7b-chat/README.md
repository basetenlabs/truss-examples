# Mistral 7B Chat

This is a [Truss](https://truss.baseten.co/) for Mistral 7B Chat. This Truss is compatible with our [bridge endpoint for OpenAI ChatCompletion users](https://docs.baseten.co/api-reference/openai).

Mistral 7B is a seven billion parameter language model released by [Mistral](https://mistral.ai/). Its instruct-tuned chat variant performs well versus other open source LLMs of the same size, and the model is capable of writing code.

## Deployment

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd mistral/mistral-7b-chat
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `mistral-7b-chat` as your working directory, you can deploy the model with:

```sh
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).


## Mistral 7B Chat API documentation

This section provides an overview of the Mistral 7B Chat model, its parameters, and how to use it. The API consists of a single route named  `predict`, which you can invoke to generate text.

### API route: `predict`

The `predict` route is the primary method for generating text based on a list of messages. It takes several parameters:

- __messages__ (required): A list of JSON objects representing a conversation.
- __stream__ (optional, default=False): A boolean determining whether the model should stream a response back. When `True`, the API returns generated text as it becomes available.
- __max_tokens__ (optional, default=512): The maximum number of tokens to return, counting input tokens. Maximum of 4096.
- __temperature__ (optional, default=1.0): Controls the randomness of the generated text. Higher values produce more diverse results, while lower values produce more deterministic results.
- __top_p__ (optional, default=0.95): The cumulative probability threshold for token sampling. The model will only consider tokens whose cumulative probability is below this threshold.
- __top_k__ (optional, default=50): The number of top tokens to consider when sampling. The model will only consider the top_k highest-probability tokens.
- __repetition_penalty__ (optional, default=1.0): Controls the model’s penalty on producing the same token sequence, with higher values discouraging repetition.
- __no_repeat_ngram_size__ (optional, default=0): The size of the n-gram that should not appear more than once in the output text.
- __use_cache__ (optional, default=True): A boolean determining whether the model should use the cache to avoid recomputing already computed hidden states during text generation.
- __do_sample__ (optional, default=True): Controls the sampling strategy during the decoding process. Setting to False results in the generation process sampling the highest probability next token (greedy decoding). Otherwise, we sample non-greedily via Top-K or Top-P sampling.

Here is an example of the `messages` the model takes as input:
```json
[
    {"role": "user", "content": "What is a mistral?"},
    {"role": "assistant", "content": "A mistral is a type of cold, dry wind that blows across the southern slopes of the Alps from the Valais region of Switzerland into the Ligurian Sea near Genoa. It is known for its strong and steady gusts, sometimes reaching up to 60 miles per hour."},
    {"role": "user", "content": "How does the mistral wind form?"}
]
```

## Example usage of the API in Python:

This model is designed to work with the OpenAI Chat Completions format. Here is an example of how to stream tokens using chat completions:

```python
from openai import OpenAI
import os

model_id = ""

client = OpenAI(
   api_key=os.environ["BASETEN_API_KEY"],
   base_url=f"https://bridge.baseten.co/{model_id}/v1"
)

res = client.chat.completions.create(
 model="mistral-7b",
 messages=[
   {"role": "user", "content": "What is a mistral?"},
   {"role": "assistant", "content": "A mistral is a type of cold, dry wind that blows across the southern slopes of the Alps from the Valais region of Switzerland into the Ligurian Sea near Genoa. It is known for its strong and steady gusts, sometimes reaching up to 60 miles per hour."},
   {"role": "user", "content": "How does the mistral wind form?"}
 ],
 temperature=0.5,
 max_tokens=50,
 top_p=0.95,
 stream=True
)

for chunk in res:
    print(chunk.choices[0].delta.content)
```

Similarly, here is a non-streaming example:

```python
from openai import OpenAI
import os

model_id = ""

client = OpenAI(
   api_key=os.environ["BASETEN_API_KEY"],
   base_url=f"https://bridge.baseten.co/{model_id}/v1"
)

res = client.chat.completions.create(
 model="mistral-7b",
 messages=[
   {"role": "user", "content": "What is a mistral?"},
   {"role": "assistant", "content": "A mistral is a type of cold, dry wind that blows across the southern slopes of the Alps from the Valais region of Switzerland into the Ligurian Sea near Genoa. It is known for its strong and steady gusts, sometimes reaching up to 60 miles per hour."},
   {"role": "user", "content": "How does the mistral wind form?"}
 ],
 temperature=0.5,
 max_tokens=50,
 top_p=0.95
)

print(res.choices[0].message.content)
```

It's not necessary to use OpenAI Chat Completions. You can also invoke the model using http requests.

Streaming Example:

```python
import requests

model_id = ""
baseten_api_key = os.environ["BASETEN_API_KEY"]

messages = [
    {"role": "user", "content": "What is a mistral?"},
    {"role": "assistant", "content": "A mistral is a type of cold, dry wind that blows across the southern slopes of the Alps from the Valais region of Switzerland into the Ligurian Sea near Genoa. It is known for its strong and steady gusts, sometimes reaching up to 60 miles per hour."},
    {"role": "user", "content": "How does the mistral wind form?"},
]
data = {
    "messages": messages,
    "stream": True,
    "max_new_tokens": 100,
    "temperature": 0.9,
    "top_p": 0.85,
    "top_k": 40,
    "repetition_penalty": 1.1,
    "no_repeat_ngram_size": 3
}

res = requests.post(
    f"https://model-{model_id}.api.baseten.co/production/predict",
    headers={"Authorization": f"Api-Key {baseten_api_key}"},
    json=data,
    stream=True
)

for content in res.iter_content():
    print(content.decode("utf-8"), end="", flush=True)
```

Non-streaming Example:

```python
import requests

model_id = ""
baseten_api_key = os.environ["BASETEN_API_KEY"]

messages = [
    {"role": "user", "content": "What is a mistral?"},
    {"role": "assistant", "content": "A mistral is a type of cold, dry wind that blows across the southern slopes of the Alps from the Valais region of Switzerland into the Ligurian Sea near Genoa. It is known for its strong and steady gusts, sometimes reaching up to 60 miles per hour."},
    {"role": "user", "content": "How does the mistral wind form?"},
]
data = {
    "messages": messages,
    "max_new_tokens": 100,
    "temperature": 0.9,
    "top_p": 0.85,
    "top_k": 40,
    "repetition_penalty": 1.1,
    "no_repeat_ngram_size": 3
}

res = requests.post(
    f"https://model-{model_id}.api.baseten.co/production/predict",
    headers={"Authorization": f"Api-Key {baseten_api_key}"},
    json=data
)

print(res.json())
```

When the model is invoked without streaming the output is a string which contains the generated text.
Here is an example of the LLM output:

```
"[INST] What is a mistral? [/INST]A mistral is a type of cold, dry wind that blows across the southern slopes of the Alps from the Valais region of Switzerland into the Ligurian Sea near Genoa. It is known for its strong and steady gusts, sometimes reaching up to 60 miles per hour.  [INST] How does the mistral wind form? [/INST]The mistral wind forms as a result of the movement of cold air from the high mountains of the Swiss Alps towards the sea. The cold air collides with the warmer air over the Mediterranean Sea, causing the cold air to rise rapidly and creating a cyclonic circulation. As the warm air rises, the cold air flows into the valley, creating a strong, steady wind known as the mistral.\n\nThe mistral is typically strongest during the winter months when the air is cold."
```
