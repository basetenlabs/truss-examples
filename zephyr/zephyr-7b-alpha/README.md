# Zephy 7B Alpha

This is a [Truss](https://truss.baseten.co/) for Zephyr 7B Alpha, an LLM optimized for chat.


## Deployment

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd zephyr-7b-alpha
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `zephyr-7b-alpha` as your working directory, you can deploy the model with:

```sh
truss push --publish
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

### Hardware notes

This seven billion parameter model is running in `float16` so that it fits on an A10G.

## Zephyr 7B Alpha API documentation

This section provides an overview of the Mistral 7B Instruct API, its parameters, and how to use it. The API consists of a single route named  `predict`, which you can invoke to generate text based on the provided prompt.

### API route: `predict`

The `predict` route is the primary method for generating text completions based on a given prompt. It takes several parameters:

- __prompt__: The input text that you want the model to generate a response for.
- __stream__ (optional, default=False): A boolean determining whether the model should stream a response back. When `True`, the API returns generated text as it becomes available.
- __max_tokens__ (optional, default=512): The maximum number of tokens to return, counting input tokens. Maximum of 4096.
- __temperature__ (optional, default=1.0): Controls the randomness of the generated text. Higher values produce more diverse results, while lower values produce more deterministic results.
- __top_p__ (optional, default=0.95): The cumulative probability threshold for token sampling. The model will only consider tokens whose cumulative probability is below this threshold.
- __top_k__ (optional, default=50): The number of top tokens to consider when sampling. The model will only consider the top_k highest-probability tokens.
- __repetition_penalty__ (optional, default=1.0): Controls the model’s penalty on producing the same token sequence, with higher values discouraging repetition.
- __no_repeat_ngram_size__ (optional, default=0): The size of the n-gram that should not appear more than once in the output text.
- __use_cache__ (optional, default=True): A boolean determining whether the model should use the cache to avoid recomputing already computed hidden states during text generation.
- __do_sample__ (optional, default=True): Controls the sampling strategy during the decoding process. Setting to False results in the generation process sampling the highest probability next token (greedy decoding). Otherwise, we sample non-greedily via Top-K or Top-P sampling.

## Example usage

```sh
truss predict -d '{"messages": [{"role": "user", "content": "What is the mistral wind?"}]}'
```

You can also invoke your model via a REST API:

```
curl -X POST " https://app.baseten.co/model_versions/YOUR_MODEL_VERSION_ID/predict" \
     -H "Content-Type: application/json" \
     -H 'Authorization: Api-Key {YOUR_API_KEY}' \
     -d '{
          {"messages": [{"role": "user", "content": "What is the mistral wind?"}]}
         }' --no-buffer
```

## Usage via the OpenAI Python

You can also invoke this model via the OpenAI Python client, simply replace the $MODEL_ID with
ID if your deployed model.

```python
client = OpenAI(
    api_key=os.environ["BASETEN_API_KEY"],
    base_url="https://bridge.baseten.co/$MODEL_ID/v1"
)

response = client.chat.completions.create(
  model="mistral-7b",
  messages=[
    {"role": "user", "content": "Who won the world series in 2020?"},
    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
    {"role": "user", "content": "Where was it played?"}
  ],
  stream=True
)

print(response)
# Streaming
for chunk in response:
    print(chunk.choices[0].delta)
```
