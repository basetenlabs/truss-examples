# Mistral-7B-Instruct-Chat Truss

This is a [Truss](https://truss.baseten.co/) for Mistral 7B Instruct. This README will walk you through how to deploy this Truss on Baseten to get your own instance of Mistral 7B Instruct.

**Warning: This example is only intended for usage on a single H100, changing your resource type for this deployment will result in unsupported behavior**

## Truss

Truss is an open-source model serving framework developed by Baseten. It allows you to develop and deploy machine learning models onto Baseten. Using Truss, you can develop a GPU model using [live-reload](https://baseten.co/blog/technical-deep-dive-truss-live-reload), package models and their associated code, create Docker containers and deploy on Baseten.

## Deploying Mistral-7B-Instruct-Chat

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd mistral/mistral-7b-instruct-chat-trt-llm-h100
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `mistral-7b-instruct-chat-trt-llm` as your working directory, you can deploy the model with:

```sh
truss push --publish
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

## Mistral 7B Instruct API documentation

This section provides an overview of the Mistral 7B Instruct API, its parameters, and how to use it. The API consists of a single route named `predict`, which you can invoke to generate text based on the provided instruction.

### API route: `predict`

This model is designed for our ChatCompletions endpoint:

- [ChatCompletions endpoint tutorial](https://www.baseten.co/blog/gpt-vs-mistral-migrate-to-open-source-llms-with-minor-code-changes/)
- [ChatCompletions endpoint reference docs](https://docs.baseten.co/api-reference/openai)

We expect requests will the following information:

- `messages` (str): The prompt you'd like to complete
- `max_tokens` (int, default: 50): The max token count. This includes the number of tokens in your prompt so if this value is less than your prompt, you'll just recieve a truncated version of the prompt.
- `beam_width` (int, default:50): The number of beams to compute. This must be 1 for this version of TRT-LLM. Inflight-batching does not support beams > 1.
- `bad_words_list` (list, default:[]): A list of words to not include in generated output.
- `stop_words_list` (list, default:[]): A list of words to stop generation upon encountering.
- `repetition_penalty` (float, defualt: 1.0): A repetition penalty to incentivize not repeating tokens.

This Truss will stream responses back. Responses will be buffered chunks of text.
