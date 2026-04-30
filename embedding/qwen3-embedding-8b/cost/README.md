# BEI (Baseten-Embeddings-Inference) with Qwen/Qwen3-Embedding-8B

This is a Deployment for BEI (Baseten-Embeddings-Inference) with Qwen/Qwen3-Embedding-8B. BEI is Baseten's solution for production-grade deployments via TensorRT-LLM for (text) embeddings, reranking models and prediction models.
With BEI you get the following benefits:
- *Lowest-latency inference* across any embedding solution (vLLM, SGlang, Infinity, TEI, Ollama)<sup>1</sup>
- *Highest-throughput inference* across any embedding solution (vLLM, SGlang, Infinity, TEI, Ollama) - thanks to XQA kernels, FP8 and dynamic batching.<sup>2</sup>
- High parallelism: up to 1400 client embeddings per second
- Cached model weights for fast vertical scaling and high availability - no Hugging Face hub dependency at runtime


# Examples:
This deployment is specifically designed for the Hugging Face model [michaelfeil/Qwen3-Embedding-8B-auto](https://huggingface.co/michaelfeil/Qwen3-Embedding-8B-auto), a re-uploaded checkpoint of the official [Qwen/Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) with an architecture string compatible with BEI's encoder build path.

Qwen3-Embedding-8B is a state-of-the-art text embedding model. It maps text into high-dimensional dense vectors used for semantic search, retrieval-augmented generation (RAG), clustering, and classification.

This model is quantized to FP8 for deployment, which is supported by Nvidia's newest GPUs e.g. H100, H100_40GB or L4. Quantization is optional, but leads to higher efficiency.

## Deployment with Truss

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`


First, clone this repository:
```sh
git clone https://github.com/basetenlabs/truss-examples.git
cd 11-embeddings-reranker-classification-tensorrt/BEI-qwen-qwen3-embedding-8b-fp8
```

With `11-embeddings-reranker-classification-tensorrt/BEI-qwen-qwen3-embedding-8b-fp8` as your working directory, you can deploy the model with the following command. Paste your Baseten API key if prompted.

```sh
truss push --publish
# prints:
# ✨ Model BEI-qwen-qwen3-embedding-8b-fp8-truss-example was successfully pushed ✨
# 🪵  View logs for your deployment at https://app.baseten.co/models/yyyyyy/logs/xxxxxx
```

## Call your model

### API-Schema (OpenAI-compatible):
POST-Route: `https://model-xxxxxx.api.baseten.co/environments/production/sync/v1/embeddings`
```json
{
  "input": ["Baseten is a fast inference provider", "Embeddings let you do semantic search."],
  "model": "qwen3-embedding-8b"
}
```

Returns:
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.0123, -0.0456, "..."]
    }
  ],
  "model": "qwen3-embedding-8b",
  "usage": {"prompt_tokens": 12, "total_tokens": 12}
}
```


### Baseten Performance Client

Read more on the [Baseten Performance Client Blog](https://www.baseten.co/blog/your-client-code-matters-10x-higher-embedding-throughput-with-python-and-rust/)


```bash
pip install baseten-performance-client
```

```python
import os
from baseten_performance_client import PerformanceClient

client = PerformanceClient(
    api_key=os.environ['BASETEN_API_KEY'],
    base_url="https://model-xxxxxx.api.baseten.co/environments/production/sync"
)

response = client.embed(
    input=["Baseten is a fast inference provider", "Embeddings let you do semantic search."],
    model="qwen3-embedding-8b"
)
print(response.data)
```

### OpenAI client library
```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["BASETEN_API_KEY"],
    base_url="https://model-xxxxxx.api.baseten.co/environments/production/sync/v1"
)

response = client.embeddings.create(
    input=["Baseten is a fast inference provider", "Embeddings let you do semantic search."],
    model="qwen3-embedding-8b"
)
print(response.data[0].embedding)
```

### Requests python library
```python
import os
import requests

headers = {
    "Authorization": f"Api-Key {os.environ['BASETEN_API_KEY']}"
}

requests.post(
    headers=headers,
    url="https://model-xxxxxx.api.baseten.co/environments/production/sync/v1/embeddings",
    json={
        "input": ["Baseten is a fast inference provider", "Embeddings let you do semantic search."],
        "model": "qwen3-embedding-8b"
    }
)
```

Important, this is different from the `predict` route that you usually call. (https://model-xxxxxx.api.baseten.co/environments/production/predict), it contains an additional `sync` before that.
The OpenAPI.json is available under https://model-xxxxxx.api.baseten.co/environments/production/sync/openapi.json for more details.

#### Advanced:
You may also use Baseten's async jobs API, which returns a request_id, which you can use to query the status of the job and get the results.

POST-Route: `https://model-xxxxxx.api.baseten.co/environments/production/async/v1/embeddings`
Read more about [Baseten's Async API here](https://docs.baseten.co/invoke/async)


## Config.yaml
By default, the following configuration is used for this deployment. This config uses `quantization_type=fp8`. This is optional, remove the `quantization_type` field or set it to `no_quant` for float16/bfloat16.

```yaml
model_metadata:
  example_model_input:
    input:
      - Baseten is a fast inference provider
      - Embeddings let you do semantic search.
    model: qwen3-embedding-8b
model_name: BEI-qwen-qwen3-embedding-8b-fp8-truss-example
python_version: py39
resources:
  accelerator: H100_40GB
  cpu: '1'
  memory: 10Gi
  use_gpu: true
trt_llm:
  build:
    base_model: encoder
    checkpoint_repository:
      repo: michaelfeil/Qwen3-Embedding-8B-auto
      revision: main
      source: HF
    max_num_tokens: 40960
    num_builder_gpus: 1
    quantization_type: fp8
  runtime:
    webserver_default_route: /v1/embeddings

```

## Support
If you have any questions or need assistance, please open an issue in this repository or contact our support team.