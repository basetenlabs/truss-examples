# Huggingface's text-embeddings-inference with Alibaba-NLP/gte-reranker-modernbert-base

This is a Deployment for Huggingface's text-embeddings-inference with Alibaba-NLP/gte-reranker-modernbert-base. TEI is huggingface's solution for (text) embeddings, reranking models and prediction models.

Supported models are tagged here: https://huggingface.co/models?other=text-embeddings-inference&sort=trending

For TEI you have to perform a manual selection of the Docker Image. We have mirrored the following images:
```
CPU	baseten/text-embeddings-inference-mirror:cpu-1.7.1
Turing (T4, ...)	baseten/text-embeddings-inference-mirror:turing-1.7.1
Ampere 80 (A100, A30)	baseten/text-embeddings-inference-mirror:1.7.1
Ampere 86 (A10, A10G, A40, ...)	baseten/text-embeddings-inference-mirror:86-1.7.1
Ada Lovelace (L4, ...)	baseten/text-embeddings-inference-mirror:89-1.7.1
Hopper (H100/H100 40GB/H200)	baseten/text-embeddings-inference-mirror:hopper-1.7.1
```

As we are deploying mostly tiny models (<1GB), we are downloading the model weights into the docker image.
For larger models, we recommend downloading the weights at runtime for faster autoscaling, as the weights don't need to go through decompression of the docker image.


# Examples:
This deployment is specifically designed for the Hugging Face model [Alibaba-NLP/gte-reranker-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base).
Suitable models can be identified by the `ForSequenceClassification` suffix in the model name. Reranker models may have at most one label, which contains the score of the reranking.

Alibaba-NLP/gte-reranker-modernbert-base  is a reranker model, used to re-rank a list of items, given a query. \nIt is frequently used in search engines, recommendation systems, and more.


## Deployment with Truss

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`


First, clone this repository:
```sh
git clone https://github.com/basetenlabs/truss-examples.git
cd 11-embeddings-reranker-classification-tensorrt/TEI-alibaba-nlp-gte-reranker-modernbert-base
```

With `11-embeddings-reranker-classification-tensorrt/TEI-alibaba-nlp-gte-reranker-modernbert-base` as your working directory, you can deploy the model with the following command. Paste your Baseten API key if prompted.

```sh
truss push --publish
# prints:
# ✨ Model TEI-alibaba-nlp-gte-reranker-modernbert-base-truss-example was successfully pushed ✨
# 🪵  View logs for your deployment at https://app.baseten.co/models/yyyyyy/logs/xxxxxx
```

## Call your model

### API-Schema:
POST-Route: `https://model-xxxxxx.api.baseten.co/environments/production/sync/rerank`:
```json
{
    "query": "What is Baseten?",
    "raw_scores": true,
    "return_text": false,
    "texts": [
        "Deep Learning is ...", "Baseten is a fast inference provider"
    ],
    "truncate": true,
    "truncation_direction": "Right"
}
```

### Baseten Performance Client

Read more on the [Baseten Performance Client Blog](https://www.baseten.co/blog/your-client-code-matters-10x-higher-embedding-throughput-with-python-and-rust/)

```python
from baseten_performance_client import PerformanceClient

client = PerformanceClient(
    api_key=os.environ['BASETEN_API_KEY'],
    base_url="https://model-xxxxxx.api.baseten.co/environments/production/sync"
)
response = client.rerank(
    query="What is Baseten?",
    texts=["Deep Learning is ...", "Baseten is a fast inference provider"],
    raw_scores=True,
    return_text=False,
    truncate=True,
)
print(response.data)
```

Sometimes, you may want to apply a custom template to the texts before reranking them and call the predict endpoint instead:

```python
from baseten_performance_client import PerformanceClient

client = PerformanceClient(
    api_key=os.environ['BASETEN_API_KEY'],
    base_url="https://model-xxxxxx.api.baseten.co/environments/production/sync"
)
def template(text: list[str]) -> list[str]:
    # Custom template function to apply to the texts
    # a popular template might be "{query}\n{document}"
    # or also chat-style templates like "User: {query}\nDocument: {document}"
    apply = lambda x: f"Custom template: {x}"
    return [apply(t) for t in text]

response = client.predict(
    inputs=template(["What is baseten? A: Baseten is a fast inference provider", "Classify this separately."]),
    raw_scores=True,
    truncate=True,
)
print(response.data)
```


### Requests python library

```python
import requests
import os

headers = {
    f"Authorization": f"Api-Key {os.environ['BASETEN_API_KEY']}"
}

requests.post(
    headers=headers,
    url="https://model-xxxxxx.api.baseten.co/environments/production/sync/rerank",
    json={
    "query": "What is Baseten?",
    "raw_scores": True,
    "return_text": False,
    "texts": [
        "Deep Learning is ...", "Baseten is a fast inference provider"
    ],
    "truncate": True,
    "truncation_direction": "Right"
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
  image: baseten/text-embeddings-inference-mirror:89-1.7.1
docker_server:
  liveness_endpoint: /health
  predict_endpoint: /rerank
  readiness_endpoint: /health
  server_port: 7997
  start_command: bash -c "truss-transfer-cli && text-embeddings-router --port 7997
    --model-id /app/model_cache/cached_model --max-client-batch-size 128 --max-concurrent-requests
    128 --max-batch-tokens 16384 --auto-truncate"
model_cache:
- ignore_patterns:
  - '*.pt'
  - '*.ckpt'
  - '*.onnx'
  repo_id: Alibaba-NLP/gte-reranker-modernbert-base
  revision: main
  use_volume: true
  volume_folder: cached_model
model_metadata:
  example_model_input:
    query: What is Baseten?
    raw_scores: true
    return_text: true
    texts:
    - Deep Learning is ...
    - Baseten is a fast inference provider
    truncate: true
    truncation_direction: Right
model_name: TEI-alibaba-nlp-gte-reranker-modernbert-base-truss-example
python_version: py39
resources:
  accelerator: L4
  cpu: '1'
  memory: 2Gi
  use_gpu: true
runtime:
  is_websocket_endpoint: false
  predict_concurrency: 32
  transport:
    kind: http

```

## Support
If you have any questions or need assistance, please open an issue in this repository or contact our support team.
