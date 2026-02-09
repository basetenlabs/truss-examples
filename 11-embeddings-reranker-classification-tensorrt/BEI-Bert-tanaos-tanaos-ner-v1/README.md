# BEI-Bert (Baseten-Embeddings-Inference-BERT) with tanaos/tanaos-NER-v1

This is a Deployment for BEI-Bert (Baseten-Embeddings-Inference-BERT) with tanaos/tanaos-NER-v1. BEI is Baseten's solution for production-grade deployments via TensorRT-LLM for (text) embeddings, reranking models and prediction models.
With BEI you get the following benefits:
- *Lowest-latency inference* across any embedding solution (vLLM, SGlang, Infinity, TEI, Ollama)<sup>1</sup>
- *Highest-throughput inference* across any embedding solution (vLLM, SGlang, Infinity, TEI, Ollama) - thanks to XQA kernels, FP8 and dynamic batching.<sup>2</sup>
- High parallelism: up to 1400 client embeddings per second
- Cached model weights for fast vertical scaling and high availability - no Hugging Face hub dependency at runtime


# Examples:
This deployment is specifically designed for the Hugging Face model [tanaos/tanaos-NER-v1](https://huggingface.co/tanaos/tanaos-NER-v1).
Suitable models can be identified by the `ForTokenClassification` suffix in the model name. NER models classify each token in the input text into entity categories (e.g., PER, ORG, LOC) or 'O' (outside any entity).

tanaos/tanaos-NER-v1  is a Named Entity Recognition (NER) model, used to identify and classify named entities in text. \nIt is frequently used for information extraction, entity linking, and document analysis. Common entities include persons, organizations, locations, dates, and more.


## Deployment with Truss

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`


First, clone this repository:
```sh
git clone https://github.com/basetenlabs/truss-examples.git
cd 11-embeddings-reranker-classification-tensorrt/BEI-Bert-tanaos-tanaos-ner-v1
```

With `11-embeddings-reranker-classification-tensorrt/BEI-Bert-tanaos-tanaos-ner-v1` as your working directory, you can deploy the model with the following command. Paste your Baseten API key if prompted.

```sh
truss push --publish
# prints:
# ✨ Model BEI-Bert-tanaos-tanaos-ner-v1-truss-example was successfully pushed ✨
# 🪵  View logs for your deployment at https://app.baseten.co/models/yyyyyy/logs/xxxxxx
```

## Call your model

### API-Schema:
POST-Route: `https://model-xxxxxx.api.baseten.co/environments/production/sync/predict_tokens`
```json
{
  "inputs": ["Apple is looking at buying U.K. startup for $1 billion"],
  "raw_scores": true,
  "truncate": true,
  "truncation_direction": "Right"
}
```

### Baseten Performance Client

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
        "raw_scores": False,
        "truncate": True,
        "truncation_direction": "Right"
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
        "raw_scores": True,
        "truncate": True,
        "truncation_direction": "Right"
    }
)
print(response.json())
```
Returns:
```json
[
  [
    {
      "token": "[CLS]",
      "token_id": 101,
      "start": 0,
      "end": 0,
      "results": {
        "O": 9.4140625,
        "B-MISC": -1.15625,
        "I-MISC": -0.859375,
        "B-PER": -1.2744141,
        "I-PER": -1.6552734,
        "B-ORG": -0.88378906,
        "I-ORG": -0.9345703,
        "B-LOC": -1.2275391,
        "I-LOC": -1.4042969
      }
    },
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
    inputs:
    - - Apple is looking at buying U.K. startup for $1 billion
    - - John works at Google in Mountain View, California
    raw_scores: true
    truncate: true
    truncation_direction: Right
model_name: BEI-Bert-tanaos-tanaos-ner-v1-truss-example
python_version: py39
resources:
  accelerator: L4
  cpu: '1'
  memory: 10Gi
  use_gpu: true
trt_llm:
  build:
    base_model: encoder_bert
    checkpoint_repository:
      repo: tanaos/tanaos-NER-v1
      revision: main
      source: HF
    max_num_tokens: 16384
  runtime:
    webserver_default_route: /rerank

```

## Support
If you have any questions or need assistance, please open an issue in this repository or contact our support team.
