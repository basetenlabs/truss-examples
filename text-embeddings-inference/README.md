# Text Embeddings Inference Truss

This is a [Truss](https://truss.baseten.co/) to deploy [text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference), a high performance embedding and reranking api.

## Deployment

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`
3. [Required for gated model] Retrieve your Hugging Face token from the [settings](https://huggingface.co/settings/tokens). Set your Hugging Face token as a Baseten secret [here](https://app.baseten.co/settings/secrets) with the key `hf_access_key`.

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples.git
cd text-embeddings-inference
```

With `text-embeddings-inference` as your working directory, you can deploy the model with the following command, paste your Baseten API key if prompted.

```sh
truss push --publish
```

## Call your model

### curl

```bash
curl -X POST https://model-xxx.api.baseten.co/development/predict \
        -H "Authorization: Api-Key YOUR_API_KEY" \
        -d '{"input": "text string"}'
```


### request python library

```python
import os
import requests

resp = requests.post(
    "https://model-xxx.api.baseten.co/environments/production/predict",
    headers={"Authorization": f"Api-Key {os.environ['BASETEN_API_KEY']}"},
    json={"input": ["text string", "second string"]},
)

print(resp.json())
```


## Support

If you have any questions or need assistance, please open an issue in this repository or contact our support team.
