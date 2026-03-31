# Qwen 7B Chat Truss

This is a [Truss](https://truss.baseten.co/) for Qwen-7B Chat. Qwen is a family of models developed by Alibaba Cloud. This LLM supports both English and Chinese.
## Truss

Truss is an open-source model serving framework developed by Baseten. It allows you to develop and deploy machine learning models onto Baseten (and other platforms like [AWS](https://truss.baseten.co/deploy/aws) or [GCP](https://truss.baseten.co/deploy/gcp). Using Truss, you can develop a GPU model using [live-reload](https://baseten.co/blog/technical-deep-dive-truss-live-reload), package models and their associated code, create Docker containers and deploy on Baseten.


## Deployment

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd qwen-7b-chat
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `qwen-7b-chat` as your working directory, you can deploy the model with:

```sh
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).


### Hardware notes

This seven billion parameter model requires an A10 GPU.

## Qwen-7B Chat API documentation

This section provides an overview of the Qwen-7B Chat model, its parameters, and how to use it. The API consists of a single route named  `predict`, which you can invoke to generate text based on the provided prompt.

### API route: `predict`

The predict route is the primary method for generating text completions based on a given prompt. It takes several parameters:

- __prompt__: The input text that you want the model to generate a response for.
- __stream__ (optional, default=False): A boolean determining whether the model should stream a response back. When `True`, the API returns generated text as it becomes available.
- __max_new_tokens__ (optional, default=512): The maximum number of tokens to return, counting input tokens. Maximum of 4096.
- __temperature__ (optional, default=0.5): Controls the randomness of the generated text. Higher values produce more diverse results, while lower values produce more deterministic results.
- __top_p__ (optional, default=0.95): The cumulative probability threshold for token sampling. The model will only consider tokens whose cumulative probability is below this threshold.
- __top_k__ (optional, default=40): The number of top tokens to consider when sampling. The model will only consider the top_k highest-probability tokens.

The API also supports passing any parameter supported by HuggingFace's `Transformers.generate`.

## Example usage

```sh
truss predict -d '{"prompt": "What is the meaning of life?", "max_new_tokens": 512}'
```

You can also invoke your model via a REST API:

```
curl -X POST " https://app.baseten.co/model_versions/YOUR_MODEL_VERSION_ID/predict" \
     -H "Content-Type: application/json" \
     -H 'Authorization: Api-Key {YOUR_API_KEY}' \
     -d '{
           "prompt": "What's the meaning of life?",
           "max_new_tokens": 512
         }'
```

### Model Output

```txt
The meaning of life is a philosophical question that has been debated throughout history. Different people have different beliefs and opinions about what the purpose of existence is, and there is no one definitive answer.

Some believe that the meaning of life is to seek happiness and fulfillment, while others think it is to serve a higher power or to fulfill a specific destiny. Some believe that life has no inherent meaning and that we must create our own purpose through our actions and experiences.

Ultimately, the meaning of life is a deeply personal and subjective concept that may vary from person to person. It is up to each individual to determine their own values and beliefs, and to create a purposeful life that aligns with those beliefs.
```
