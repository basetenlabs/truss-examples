# CodeGen mono 2B Truss

This is an implementation of the Salesforce [CodeGen](https://github.com/salesforce/CodeGen) model. The model
was trained using the `mono` dataset and this version is the 2B parameter model. This model is specialized for Python
code production.
## Deploying CodeGen mono 2B

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd codegen-mono-2b-truss
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `codegen-mono-2b-truss` as your working directory, you can deploy the model with:

```sh
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

## CodeGen mono 2B API documentation

### Input

The input should be a dictionary. It supports the following keys:

* `prompt` - the natural language or code prompt desired to generate.
* `max_length` - optional, the maximum length for generation, maxes out at 128 tokens
* `temperature` - optional, the temperature for the generator. defaults to 0.2
* `top_p` - optional, the top_p for the generator. defaults to 0.95

For example:

```json
{
    "prompt": "def fibonacci(n):"
}
```

### Output

The result will be a dictionary that will have the following keys:

* `completion` - the full generation of the model
* `truncation` - a heuristically truncated segment of the code
* `context` - the context provided to the model

For example:

```json
{
    "completion": "code for fibonacci function\r\ndef fib(n):\r\n...",
    "prompt": "code for fibonacci",
    "truncation": " function\r\ndef fib(n):\r\n..."
}
```

## Example usage

```sh
truss predict -d '{"prompt": "code for fibonacci"}'
```

You can also invoke this model on Baseten with the following cURL command (just fill in the model version ID and API Key):

```bash
$ curl -X POST https://app.baseten.co/model_versions/{MODEL_VERSION_ID}/predict
       -H 'Authorization: Api-Key {YOUR_API_KEY}'
       -d '{"prompt": "code for fibonacci"}'
```
