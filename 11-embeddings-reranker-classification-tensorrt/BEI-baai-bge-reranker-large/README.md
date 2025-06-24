# BEI (Baseten-Embeddings-Inference) with BAAI/bge-reranker-large

This is a Deployment for BEI (Baseten-Embeddings-Inference) with BAAI/bge-reranker-large. BEI is Baseten's solution for production-grade deployments via TensorRT-LLM for (text) embeddings, reranking models and prediction models.
With BEI you get the following benefits:
- *Lowest-latency inference* across any embedding solution (vLLM, SGlang, Infinity, TEI, Ollama)<sup>1</sup>
- *Highest-throughput inference* across any embedding solution (vLLM, SGlang, Infinity, TEI, Ollama) - thanks to XQA kernels, FP8 and dynamic batching.<sup>2</sup>
- High parallelism: up to 1400 client embeddings per second
- Cached model weights for fast vertical scaling and high availability - no Hugging Face hub dependency at runtime


# Examples:
This deployment is specifically designed for the Hugging Face model [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large).
Suitable models can be identified by the `ForSequenceClassification` suffix in the model name. Reranker models may have at most one label, which contains the score of the reranking.

BAAI/bge-reranker-large  is a reranker model, used to re-rank a list of items, given a query. \nIt is frequently used in search engines, recommendation systems, and more.


## Deployment with Truss

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`


First, clone this repository:
```sh
git clone https://github.com/basetenlabs/truss-examples.git
cd 11-embeddings-reranker-classification-tensorrt/BEI-baai-bge-reranker-large
```

With `11-embeddings-reranker-classification-tensorrt/BEI-baai-bge-reranker-large` as your working directory, you can deploy the model with the following command. Paste your Baseten API key if prompted.

```sh
truss push --publish
# prints:
# ✨ Model BEI-baai-bge-reranker-large-truss-example was successfully pushed ✨
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
model_name: BEI-baai-bge-reranker-large-truss-example
python_version: py39
resources:
  accelerator: L4
  cpu: '1'
  memory: 10Gi
  use_gpu: true
trt_llm:
  build:
    base_model: encoder
    checkpoint_repository:
      repo: BAAI/bge-reranker-large
      revision: main
      source: HF
    max_num_tokens: 16384
  runtime:
    webserver_default_route: /rerank
  version_overrides:
    bei_version: 0.0.25-b200-dev-v4
    engine_builder_version: 0.20.0.dev1

```

## Support
If you have any questions or need assistance, please open an issue in this repository or contact our support team.
