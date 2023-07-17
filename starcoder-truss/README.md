# Starcoder Truss

This is a [Truss](https://truss.baseten.co/) for Starcoder. Starcoder is an open-source language model trained specifically
for code auto-completions. It was trained on text from over 80 programming languages. Check out more info about this model
[here](https://huggingface.co/bigcode/starcoder).

Before deploying this model, you'll need to:

1. Accept the terms of service of the Starcoder model [here](https://huggingface.co/bigcode/starcoder).
2. Retrieve your Huggingface token from the [settings](https://huggingface.co/settings/tokens).
3. Set your Huggingface token as a Baseten secret [here](https://app.baseten.co/settings/secrets) with the key `hf_api_key`. Note that you will *not* be able to successfully deploy Starcoder without doing this.

## Deploying Starcoder 

To deploy the Starcoder Truss, you'll need to follow these steps:

1. __Prerequisites__: Make sure you have a Baseten account and API key. You can sign up for a Baseten account [here](https://app.baseten.co/signup).

2. __Install Truss and the Baseten Python client__: If you haven't already, install the Baseten Python client and Truss in your development environment using:
```
pip install --upgrade baseten truss
```

3. __Load the Starcoder Truss__: Assuming you've cloned this repo, spin up an IPython shell and load the Truss into memory:

Note this assumes that you started the ipython shell from root of the repo.

```
import truss

starcoder_truss = truss.load(".")
```

4. __Log in to Baseten__: Log in to your Baseten account using your API key (key found [here](https://app.baseten.co/settings/account/api_keys)):
```
import baseten

baseten.login("PASTE_API_KEY_HERE")
```

5. __Deploy the Starcoder Truss__: Deploy the Starcoder Truss to Baseten with the following command:
```
baseten.deploy(starcoder_truss)
```

Once your Truss is deployed, you can start using the Starcoder model through the Baseten platform! Navigate to the Baseten UI to watch the model build and deploy and invoke it via the REST API.

## Starcoder API documentation

### Input

This deployment of Starcoder takes a dictionary as input, which requires the following key:

* `prompt` - the prompt for code auto-completion

It also supports all parameters detailed in the transformers [GenerationConfig](https://huggingface.co/docs/transformers/v4.29.1/en/main_classes/text_generation#transformers.GenerationConfig).

### Output

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

## Example usage

```
curl -X POST https://app.baseten.co/models/EqwKvqa/predict \
  -H 'Authorization: Api-Key {YOUR_API_KEY}' \
  -d '{"prompt": "def compute_fib(n):"}'
```

