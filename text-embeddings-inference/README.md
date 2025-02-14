# Notice
This section has moved to [jina-ai/jina-embeddings-v2-base-en-TEI](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt) with an overview over fast embeddings.

# Text Embeddings Inference Truss

This is a Trussless Customer Server example to deploy [text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference), a high performance server that handles text-embeddings, ranranking and classification models as api.

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

## Performance Optimization:

The config.yaml contains a couple of variables that can be tuned, depending on:
- which GPU is used
- which model is deployed
- how many concurrent requests users are sending

The deployment example is for Bert-large and a Nvidia-L4. Bert-large has a maxiumum sequence length of 512 tokens per sentence.
For Bert-large architecture & the L4, there are marginal gains above a batch-size of 16000 tokens.

### Concurrent requests
```
--max-concurrent-requests 40
# and
runtime:
  predict_concurrency : 40
```
The following set the number of parallel `post` requests.
In this case we allow 40 parallel requests to be handled per replica & should allow to batch requests from multiple users together, reaching high token counts. Potentially 40 single parallel requests with one sequence each could fully utilize the GPU. `1*40*512=20480`


### Tokens per batch
```
--max-batch-tokens 32768
```

This number of total tokens in a batch. For embedding models, this will determine the VRAM usage.
As most of TEI's models are implemented with `nested` attention implementation, `32768 tokens` could mean `64 sentence with 512 tokens` or `512 sentences with 64 tokens`. While the first will take slightly longer to compute, the peak VRAM usage will stay roughly the same. For `llama` or `mistral` based `7b` embedding models, we recommend setting it a lower setting e.g.
```
--max-batch-tokens 8192
```

### Client batch size
```
--max-client-batch-size 32
```
Client match size determines the number of sentences in a single request.
Increase if clients cannot send multiple concurrent requests, or if clients require to larger requests size.

### Endpoint, Model Selection, and OpenAPI
Change to /rerank or /predict if you want to use the rerank or predict endpoint.
Embedding model.
Example supported models: https://huggingface.co/models?pipeline_tag=feature-extraction&other=text-embeddings-inference&sort=trending
```yaml
  predict_endpoint: /v1/embeddings
```
Rerank model.
Example models https://huggingface.co/models?pipeline_tag=text-classification&other=text-embeddings-inference&sort=trending
```yaml
  predict_endpoint: /rerank
```
Classification model:
Example classification model: https://huggingface.co/SamLowe/roberta-base-go_emotions
```yaml
  predict_endpoint: /predict
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
