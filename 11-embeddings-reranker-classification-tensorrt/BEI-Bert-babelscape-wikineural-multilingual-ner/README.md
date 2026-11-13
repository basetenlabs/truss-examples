# BEI-Bert (Baseten-Embeddings-Inference-BERT) with Babelscape/wikineural-multilingual-ner

This is a Deployment for BEI-Bert (Baseten-Embeddings-Inference-BERT) with Babelscape/wikineural-multilingual-ner. BEI is Baseten's solution for production-grade deployments via TensorRT-LLM for (text) embeddings, reranking models and prediction models.
With BEI you get the following benefits:
- *Lowest-latency inference* across any embedding solution (vLLM, SGlang, Infinity, TEI, Ollama)<sup>1</sup>
- *Highest-throughput inference* across any embedding solution (vLLM, SGlang, Infinity, TEI, Ollama) - thanks to XQA kernels, FP8 and dynamic batching.<sup>2</sup>
- High parallelism: up to 1400 client embeddings per second
- Cached model weights for fast vertical scaling and high availability - no Hugging Face hub dependency at runtime


# Examples:
This deployment is specifically designed for the Hugging Face model [Babelscape/wikineural-multilingual-ner](https://huggingface.co/Babelscape/wikineural-multilingual-ner).
Suitable models can be identified by the `ForTokenClassification` suffix in the model name. NER models classify each token in the input text into entity categories (e.g., PER, ORG, LOC) or 'O' (outside any entity).

Babelscape/wikineural-multilingual-ner  is a Named Entity Recognition (NER) model, used to identify and classify named entities in text. \nIt is frequently used for information extraction, entity linking, and document analysis. Common entities include persons, organizations, locations, dates, and more.


## Deployment with Truss

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`


First, clone this repository:
```sh
git clone https://github.com/basetenlabs/truss-examples.git
cd 11-embeddings-reranker-classification-tensorrt/BEI-Bert-babelscape-wikineural-multilingual-ner
```

With `11-embeddings-reranker-classification-tensorrt/BEI-Bert-babelscape-wikineural-multilingual-ner` as your working directory, you can deploy the model with the following command. Paste your Baseten API key if prompted.

```sh
truss push --publish
# prints:
# ✨ Model BEI-Bert-babelscape-wikineural-multilingual-ner-truss-example was successfully pushed ✨
# 🪵  View logs for your deployment at https://app.baseten.co/models/yyyyyy/logs/xxxxxx
```

## Call your model

### API-Schema:
POST-Route: `https://model-xxxxxx.api.baseten.co/environments/production/sync/predict_tokens`
```json
{
  "inputs": ["Apple is looking at buying U.K. startup for $1 billion"],
  "truncate": true,
  "raw_scores": false,
  "aggregation_strategy": "max"
}
```
- `aggregation_strategy`: Controls how sub-word tokens are aggregated into entities. One of `"none"`, `"simple"`, `"first"`, `"average"`, `"max"`. Use `"none"` to get per-token results without aggregation.
- `raw_scores`: When `true`, returns raw logit scores for all labels per token. When `false`, returns only the top predicted label with its probability.

### Baseten Performance Client (Recommended)

```bash
pip install baseten-performance-client
```

```python
from baseten_performance_client import PerformanceClient

client = PerformanceClient(
    api_key=os.environ['BASETEN_API_KEY'],
    base_url="https://model-xxxxxx.api.baseten.co/environments/production/sync"
)

response = client.batch_post(
    route="/predict_tokens",
    payloads=[{
        "inputs": [["Apple is looking at buying U.K. startup for $1 billion"]],
        "truncate": True,
        "raw_scores": False,
        "aggregation_strategy": "max"
    }]
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

response = requests.post(
    headers=headers,
    url="https://model-xxxxxx.api.baseten.co/environments/production/sync/predict_tokens",
    json={
        "inputs": [["Apple is looking at buying U.K. startup for $1 billion"]],
        "truncate": True,
        "raw_scores": False,
        "aggregation_strategy": "max"
    }
)
print(response.json())
```
Returns (with `aggregation_strategy: "max"` and `raw_scores: false`):
```json
[
  [
    {
      "token": "Apple",
      "token_id": 0,
      "start": 0,
      "end": 5,
      "results": {
        "ORG": 0.9975586
      }
    },
    {
      "token": "U.K.",
      "token_id": 0,
      "start": 27,
      "end": 31,
      "results": {
        "LOC": 0.9980469
      }
    }
  ]
]
```
With `raw_scores: true` and `aggregation_strategy: "none"`, the response includes all label scores per sub-word token:
```json
[
  [
    {
      "token": "Apple",
      "token_id": 6207,
      "start": 0,
      "end": 5,
      "results": {
        "B-ORG": 6.7578125,
        "O": -1.7929688,
        "B-LOC": 0.6015625,
        "B-MISC": 0.2467041,
        "B-PER": 0.17675781,
        "I-ORG": -0.6484375,
        "I-MISC": -1.9873047,
        "I-LOC": -1.3808594,
        "I-PER": -2.21875
      }
    }
  ]
]
```
Important, this uses the `predict_tokens` endpoint for token-level classification. The OpenAPI.json is available under https://model-xxxxxx.api.baseten.co/environments/production/sync/openapi.json for more details.

#### Advanced:
You may also use Baseten's async jobs API, which returns a request_id, which you can use to query the status of the job and get the results.

POST-Route: `https://model-xxxxxx.api.baseten.co/environments/production/async/predict_tokens`
Read more about [Baseten's Async API here](https://docs.baseten.co/invoke/async)

### OpenAI compatible client library
OpenAI does not have a NER endpoint, therefore no client library is available.


## Config.yaml
By default, the following configuration is used for this deployment.

```yaml
model_metadata:
  example_model_input:
    aggregation_strategy: max
    inputs:
    - Apple is looking at buying U.K. startup for $1 billion
    - John works at Google in Mountain View, California
    raw_scores: false
    truncate: true
model_name: BEI-Bert-babelscape-wikineural-multilingual-ner-truss-example
python_version: py39
resources:
  accelerator: L4
  cpu: '1'
  memory: 10Gi
  use_gpu: true
runtime:
  health_checks:
    restart_threshold_seconds: 30
    startup_threshold_seconds: 1800
    stop_traffic_threshold_seconds: 30
  is_websocket_endpoint: false
  transport:
    kind: http
trt_llm:
  build:
    base_model: encoder_bert
    checkpoint_repository:
      repo: Babelscape/wikineural-multilingual-ner
      revision: main
      source: HF
    max_num_tokens: 16384
  runtime:
    webserver_default_route: /predict_tokens

```

## Support
If you have any questions or need assistance, please open an issue in this repository or contact our support team.
