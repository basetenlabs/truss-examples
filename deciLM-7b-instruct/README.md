# Deci LM 7B Instruct

This is a [Truss](https://truss.baseten.co/) for [DeciLM 7B Instruct](https://huggingface.co/Deci/DeciLM-7B-instruct). This LLM is known for having a 4X higher throughtput(without sacraficing on accuracy) than other LLM's such as Mistral 7B.


## Deploying

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd deciLM-7b-instruct
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `deciLM-7b-instruct` as your working directory, you can deploy the model with:

```sh
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

### Hardware notes

This seven billion parameter model is running in `float16` so that it fits on an A10G.

## DeciLM 7B API documentation

This section provides an overview of the API, its parameters, and how to use it. The API consists of a single route named  `predict`, which you can invoke to generate text based on the provided prompt.

### API route: `predict`

The predict route is the primary method for generating text completions based on a given prompt. It takes several parameters:

- __prompt__: The input text that you want the model to generate a response for.
- __max_new_tokens__ (optional, default=512): The maximum number of tokens to return, counting input tokens. Maximum of 4096.
- __temperature__ (optional, default=1.0): Controls the randomness of the generated text. Higher values produce more diverse results, while lower values produce more deterministic results.
- __top_p__ (optional, default=0.95): The cumulative probability threshold for token sampling. The model will only consider tokens whose cumulative probability is below this threshold.
- __top_k__ (optional, default=50): The number of top tokens to consider when sampling. The model will only consider the top_k highest-probability tokens.
- __stream__ (optional, default=False): Allows you to stream each word as it gets generated instead of waiting for the entire generation to complete

## Example usage

Here is an example of invoking the model using Python with `stream` enabled:
```python
data = {"prompt": "What is the meaning of life?", "stream": True}
headers = {"Authorization": "Api-Key <BASETEN-API-KEY>"}
res = requests.post("https://model-<model-id>.api.baseten.co/development/predict", headers=headers, json=data, stream=True)
res.raise_for_status()

for word in res:
    print(word.decode("utf-8"))
```


You can also invoke your model via a REST API:

```sh
curl -X POST "https://model-<model-id>.api.baseten.co/development/predict" \
     -H "Content-Type: application/json" \
     -H 'Authorization: Api-Key {YOUR_API_KEY}' \
     -d '{
           "prompt": "What is the meaning of life?"
           "max_new_tokens": 50
         }'
```

### Example output

```
prompt: What is the meaning of life?
```

```
It's amazing that the answer to those questions have been written in stone for over 2000 years (although a couple of the authors may have been paid by Big Pharma). It's amazing how close to the surface the answers have been (although another author may have hidden them in a tree). It's amazing that the answers have been given to us in a way that they must be read, deciphered and interpreted by a few people (the author, his editor, the reviewer, some of his employees, some of his children, his grandchildren, the descendants of his children and grandchildren, the scientists, the astrologists, the theologians, the philosophers, the scientists and philosophers who are alive 2000 years later, the people who read them, the descendants of those people, the scientists and philosophers who live 2000 years after our descendants, and so on).
I will go ahead and give you the answers:
"This is all there is, and there isn't any of it. It's just an illusion. It will all be over soon."
Yes, the answers can also be read to mean:
"Just don't ask any more stupid questions."
If anyone can prove that any part of those answers is wrong, then I will award him/her the Nobel Prize.
```
