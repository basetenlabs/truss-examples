# Huggingface's text-embeddings-inference with Alibaba-NLP/gte-multilingual-reranker-base

This is a Deployment for Huggingface's text-embeddings-inference with Alibaba-NLP/gte-multilingual-reranker-base. TEI is huggingface's solution for (text) embeddings, reranking models and prediction models.

Supported models are tagged here: https://huggingface.co/models?other=text-embeddings-inference&sort=trending


For TEI you have to perform a manual selection of the Docker Image. We have mirrored the following images:
```
CPU	baseten/text-embeddings-inference-mirror:cpu-1.6
Turing (T4, ...)	baseten/text-embeddings-inference-mirror:turing-1.6
Ampere 80 (A100, A30)	baseten/text-embeddings-inference-mirror:1.6
Ampere 86 (A10, A10G, A40, ...)	baseten/text-embeddings-inference-mirror:86-1.6
Ada Lovelace (L4, ...)	baseten/text-embeddings-inference-mirror:89-1.6
Hopper (H100/H100 40GB/H200)	baseten/text-embeddings-inference-mirror:hopper-1.6
```


# Examples:
This deployment is specifically designed for the Hugging Face model [Alibaba-NLP/gte-multilingual-reranker-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-reranker-base).
Suitable models can be identified by the `ForSequenceClassification` suffix in the model name. Reranker models may have at most one label, which contains the score of the reranking.

Alibaba-NLP/gte-multilingual-reranker-base  is a reranker model, used to re-rank a list of items, given a query. \nIt is frequently used in search engines, recommendation systems, and more.


## Deployment with Truss

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`


First, clone this repository:
```sh
git clone https://github.com/basetenlabs/truss-examples.git
cd 11-embeddings-reranker-classification-tensorrt/TEI-alibaba-nlp-gte-multilingual-reranker-base
```

With `11-embeddings-reranker-classification-tensorrt/TEI-alibaba-nlp-gte-multilingual-reranker-base` as your working directory, you can deploy the model with the following command. Paste your Baseten API key if prompted.

```sh
truss push --publish
# prints:
# ✨ Model TEI-alibaba-nlp-gte-multilingual-reranker-base-truss-example was successfully pushed ✨
# 🪵  View logs for your deployment at https://app.baseten.co/models/yyyyyy/logs/xxxxxx
```

## Call your model

### API-Schema:
POST-Route: `https://model-xxxxxx.api.baseten.co/environments/production/sync/rerank`:
```json
{
  "query": "What is Baseten?",
  "raw_scores": false,
  "return_text": false,
  "texts": [
    "Deep Learning is ...", "Baseten is a fast inference provider"
  ],
  "truncate": false,
  "truncation_direction": "right"
}
```

Returns:
```json
[
  {
    "index": 0,
    "score": 1,
    "text": "Deep Learning is ..."
  }
]
```
The OpenAPI.json is available under https://model-xxxxxx.api.baseten.co/environments/production/sync/openapi.json for more details.

#### Advanced:
You may also use Baseten's async jobs API, which returns a request_id, which you can use to query the status of the job and get the results.

POST-Route: `https://model-xxxxxx.api.baseten.co/environments/production/async/rerank`
Read more about [Baseten's Async API here](https://docs.baseten.co/invoke/async)

### OpenAI compatible client library
OpenAI.com does not have a rerank endpoint, therefore no client library is available.


## Config.yaml
By default, the following configuration is used for this deployment.

```yaml
base_image:
  image: baseten/text-embeddings-inference-mirror:89-1.6
build_commands:
- 'git clone https://huggingface.co/Alibaba-NLP/gte-multilingual-reranker-base /data/local-model
  # optional step to download the weights of the model into the image, otherwise specify
  the --model-id Alibaba-NLP/gte-multilingual-reranker-base directly `start_command`'
docker_server:
  liveness_endpoint: /health
  predict_endpoint: /rerank
  readiness_endpoint: /health
  server_port: 7997
  start_command: text-embeddings-router --port 7997 --model-id /data/local-model --max-client-batch-size
    128 --max-concurrent-requests 40 --max-batch-tokens 16384
environment_variables: {}
external_package_dirs: []
model_metadata:
  example_model_input:
    input: This redirects to the embedding endpoint. Use the /sync API to reach /rerank
model_name: TEI-alibaba-nlp-gte-multilingual-reranker-base-truss-example
python_version: py39
requirements: []
resources:
  accelerator: L4
  cpu: '1'
  memory: 2Gi
  use_gpu: true
runtime:
  predict_concurrency: 40
secrets: {}
system_packages: []

```

## Support
If you have any questions or need assistance, please open an issue in this repository or contact our support team.
