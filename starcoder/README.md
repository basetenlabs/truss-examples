# Starcoder Truss

This is a [Truss](https://truss.baseten.co/) for Starcoder. Starcoder is an open-source language model trained specifically
for code auto-completions. It was trained on text from over 80 programming languages. Check out more info about this model
[here](https://huggingface.co/bigcode/starcoder).

Before deploying this model, you'll need to:

1. Accept the terms of service of the Starcoder model [here](https://huggingface.co/bigcode/starcoder).
2. Retrieve your Hugging Face token from the [settings](https://huggingface.co/settings/tokens).
3. Set your Hugging Face token as a Baseten secret [here](https://app.baseten.co/settings/secrets) with the key `hf_access_key`. Note that you will *not* be able to successfully deploy Starcoder without doing this.

## Deploying Starcoder 

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `starcoder-truss` as your working directory, you can deploy the model with:

```
truss push --trusted
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

## Invoking Starcoder

To invoke Starcoder, run:

```sh
truss predict '{"prompt": "def compute_fib(n):"}'
```

You can also invoke Starcoder via an API endpoint:

```
curl -X POST https://app.baseten.co/models/EqwKvqa/predict \
  -H 'Authorization: Api-Key {YOUR_API_KEY}' \
  -d '{"prompt": "def compute_fib(n):"}'
```

### Starcoder API documentation

#### Input

This deployment of Starcoder takes a dictionary as input, which requires the following key:

* `prompt` - the prompt for code auto-completion

It also supports all parameters detailed in the transformers [GenerationConfig](https://huggingface.co/docs/transformers/v4.29.1/en/main_classes/text_generation#transformers.GenerationConfig).

#### Output

The result will be a dictionary containing:

* `status` - either `success` or `failed`
* `data` - dictionary containing keys `completion`, which is the model result, and `prompt`, which is the prompt from the input.
* `message` - will contain details in the case of errors

```json
{"status": "success",
 "data": {"completion": "code for fibonacci sequence: '))\n\ndef fibonacci(n):\n    if n == 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)\n\nprint(fibonacci(n))\n",
  "prompt": "code for fib"},
 "message": null}
```
