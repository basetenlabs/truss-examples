# BEI (Baseten-Embeddings-Inference) with Qwen/Qwen3-Reranker-8B

This is a Deployment for BEI (Baseten-Embeddings-Inference) with Qwen/Qwen3-Reranker-8B. BEI is Baseten's solution for production-grade deployments via TensorRT-LLM for (text) embeddings, reranking models and prediction models.
With BEI you get the following benefits:
- *Lowest-latency inference* across any embedding solution (vLLM, SGlang, Infinity, TEI, Ollama)<sup>1</sup>
- *Highest-throughput inference* across any embedding solution (vLLM, SGlang, Infinity, TEI, Ollama) - thanks to XQA kernels, FP8 and dynamic batching.<sup>2</sup>
- High parallelism: up to 1400 client embeddings per second
- Cached model weights for fast vertical scaling and high availability - no Hugging Face hub dependency at runtime


# Examples:
This deployment is specifically designed for the Hugging Face model [michaelfeil/Qwen3-Reranker-8B-seq](https://huggingface.co/michaelfeil/Qwen3-Reranker-8B-seq).
Suitable models can be identified by the `ForSequenceClassification` suffix in the model name. Prediction models may have one or more labels, which are returned with the prediction.

michaelfeil/Qwen3-Reranker-8B-seq  is a text-classification model, used to classify a text into a category. \nIt is frequently used in sentiment analysis, spam detection, and more. It's also used for deployment of chat rating models, e.g. RLHF reward models or toxicity detection models.


## Deployment with Truss

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`


First, clone this repository:
```sh
git clone https://github.com/basetenlabs/truss-examples.git
cd 11-embeddings-reranker-classification-tensorrt/BEI-qwen-qwen3-reranker-8b-fp4
```

With `11-embeddings-reranker-classification-tensorrt/BEI-qwen-qwen3-reranker-8b-fp4` as your working directory, you can deploy the model with the following command. Paste your Baseten API key if prompted.

```sh
truss push --publish
# prints:
# ✨ Model BEI-qwen-qwen3-reranker-8b-fp4-truss-example was successfully pushed ✨
# 🪵  View logs for your deployment at https://app.baseten.co/models/yyyyyy/logs/xxxxxx
```

## Call your model

### API-Schema:
POST-Route: `https://model-xxxxxx.api.baseten.co/environments/production/sync/predict`
```json
{
  "inputs": "Baseten is a fast inference provider",
  "raw_scores": true,
  "truncate": true,
  "truncation_direction": "Right"
}
```


### Baseten Performance Client

Read more on the [Baseten Performance Client Blog](https://www.baseten.co/blog/your-client-code-matters-10x-higher-embedding-throughput-with-python-and-rust/)


```bash
pip install baseten-performance-client
```

```python
from baseten_performance_client import PerformanceClient

client = PerformanceClient(
    api_key=os.environ['BASETEN_API_KEY'],
    base_url="https://model-xxxxxx.api.baseten.co/environments/production/sync"
)
def template(text: list[str]) -> list[str]:
    apply = lambda x: f"Custom template: {x}"
    return [apply(t) for t in text]

response = client.predict(
    inputs=template(["Baseten is a fast inference provider", "Classify this separately."]),
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
    url="https://model-xxxxxx.api.baseten.co/environments/production/sync/predict",
    json={
        "inputs": [["Baseten is a fast inference provider", ["classify this separately."]],
        "raw_scores": True,
        "truncate": True,
        "truncation_direction": "Right"
    }
)
```
Returns:
```json
[
  [
    {
        "label": "excitement",
        "score": 0.99
    }
  ],
  [
    {
        "label": "excitement",
        "score": 0.01
    }
  ]
]
```
Important, this is different from the `predict` route that you usually call. (https://model-xxxxxx.api.baseten.co/environments/production/predict), it contains an additional `sync` before that.
The OpenAPI.json is available under https://model-xxxxxx.api.baseten.co/environments/production/sync/openapi.json for more details.

#### Advanced:
You may also use Baseten's async jobs API, which returns a request_id, which you can use to query the status of the job and get the results.

POST-Route: `https://model-xxxxxx.api.baseten.co/environments/production/async/predict`
Read more about [Baseten's Async API here](https://docs.baseten.co/invoke/async)

### OpenAI compatible client library
OpenAI does not have a classification endpoint, therefore no client library is available.


## Config.yaml
By default, the following configuration is used for this deployment. This config uses `quantization_type=fp4`. This is optional, remove the `quantization_type` field or set it to `no_quant` for float16/bfloat16.

```yaml
model_metadata:
  example_model_input:
    inputs:
    - - Baseten is a fast inference provider
    - - Classify this separately.
    raw_scores: true
    truncate: true
    truncation_direction: Right
model_name: BEI-qwen-qwen3-reranker-8b-fp4-truss-example
python_version: py39
resources:
  accelerator: B200
  cpu: '1'
  memory: 10Gi
  use_gpu: true
trt_llm:
  build:
    base_model: encoder
    checkpoint_repository:
      repo: michaelfeil/Qwen3-Reranker-8B-seq
      revision: main
      source: HF
    max_num_tokens: 40960
    num_builder_gpus: 1
    quantization_type: fp4
  runtime:
    webserver_default_route: /predict

```

## Support
If you have any questions or need assistance, please open an issue in this repository or contact our support team.
