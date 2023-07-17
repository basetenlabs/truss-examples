# CodeGen mono 2B Truss

This is an implementation of the Salesforce [CodeGen](https://github.com/salesforce/CodeGen) model. The model
was trained using the `mono` dataset and this version is the 2B parameter model. This model is specialized for Python
code production.
## Deploying CodeGen mono 2B

To deploy the CodeGen mono 2B Truss, you'll need to follow these steps:

1. __Prerequisites__: Make sure you have a Baseten account and API key. You can sign up for a Baseten account [here](https://app.baseten.co/signup).

2. __Install Truss and the Baseten Python client__: If you haven't already, install the Baseten Python client and Truss in your development environment using:
```
pip install --upgrade baseten truss
```

3. __Load the CodeGen mono 2B Truss__: Assuming you've cloned this repo, spin up an IPython shell and load the Truss into memory:
```
import truss

codegen_mono_truss = truss.load("path/to/codegen_mono_truss")
```

4. __Log in to Baseten__: Log in to your Baseten account using your API key (key found [here](https://app.baseten.co/settings/account/api_keys)):
```
import baseten

baseten.login("PASTE_API_KEY_HERE")
```

5. __Deploy the CodeGen mono 2B Truss__: Deploy the CodeGen mono 2B Truss to Baseten with the following command:
```
baseten.deploy(codegen_mono_truss)
```

Once your Truss is deployed, you can start using the CodeGen mono 2B model through the Baseten platform! Navigate to the Baseten UI to watch the model build and deploy and invoke it via the REST API.

## CodeGen mono 2B API documentation

### Input

The input should be a list of dictionaries must have a key `context` which represents the prompt for generation to the
model. It supports the following keys:

* `prompt` - the natural language or code prompt desired to generate.
* `max_length` - optional, the maximum length for generation, maxes out at 128 tokens
* `temperature` - optional, the temperature for the generator. defaults to 0.2
* `top_p` - optional, the top_p for the generator. defaults to 0.95

For example:

```json
[{
    "prompt": "def fibonacci(n):"
}]
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

You can invoke this model on Baseten with the following cURL command (just fill in the model version ID and API Key):

```bash
$ curl -X POST https://app.baseten.co/model_versions/{MODEL_VERSION_ID}/predict
       -H 'Authorization: Api-Key {YOUR_API_KEY}'
       -d '{"prompt": "code for fibonacci"}'
```
