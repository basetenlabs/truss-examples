# Huggingface's text-embeddings-inference with mixedbread-ai/mxbai-embed-large-v1-embedding

This is a Deployment for Huggingface's text-embeddings-inference with mixedbread-ai/mxbai-embed-large-v1-embedding. TEI is huggingface's solution for (text) embeddings, reranking models and prediction models.

Supported models are tagged here: https://huggingface.co/models?other=text-embeddings-inference&sort=trending

For TEI you have to perform a manual selection of the Docker Image. We have mirrored the following images:
```
CPU	baseten/text-embeddings-inference-mirror:cpu-1.7.2
Turing (T4, ...)	baseten/text-embeddings-inference-mirror:turing-1.7.2
Ampere 80 (A100, A30)	baseten/text-embeddings-inference-mirror:1.7.2
Ampere 86 (A10, A10G, A40, ...)	baseten/text-embeddings-inference-mirror:86-1.7.2
Ada Lovelace (L4, ...)	baseten/text-embeddings-inference-mirror:89-1.7.2
Hopper (H100/H100 40GB/H200)	baseten/text-embeddings-inference-mirror:hopper-1.7.2
```

As we are deploying mostly tiny models (<1GB), we are downloading the model weights into the docker image.
For larger models, we recommend downloading the weights at runtime for faster autoscaling, as the weights don't need to go through decompression of the docker image.


# Examples:
This deployment is specifically designed for the Hugging Face model [mixedbread-ai/mxbai-embed-large-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1).
Suitable models need to have the configurations of the `sentence-transformers` library, which are used for embeddings. Such repos contain e.g. a `sbert_config.json` or a `1_Pooling/config.json` file besides the fast-tokenizer and the safetensors file.

mixedbread-ai/mxbai-embed-large-v1  is a text-embeddings model, producing a 1D embeddings vector, given an input.
It's frequently used for downstream tasks like clustering, used with vector databases.


## Deployment with Truss

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`


First, clone this repository:
```sh
git clone https://github.com/basetenlabs/truss-examples.git
cd 11-embeddings-reranker-classification-tensorrt/TEI-mixedbread-ai-mxbai-embed-large-v1-embedding
```

With `11-embeddings-reranker-classification-tensorrt/TEI-mixedbread-ai-mxbai-embed-large-v1-embedding` as your working directory, you can deploy the model with the following command. Paste your Baseten API key if prompted.

```sh
truss push --publish
# prints:
# ✨ Model TEI-mixedbread-ai-mxbai-embed-large-v1-embedding-truss-example was successfully pushed ✨
# 🪵  View logs for your deployment at https://app.baseten.co/models/yyyyyy/logs/xxxxxx
```

## Call your model

### API-Schema:
POST-Route: `https://model-xxxxxx.api.baseten.co/environments/production/sync/v1/embeddings`
```json
{
  "encoding_format": "float", # or base64
  "input": "string", # can be list of strings for multiple embeddings
  "model": "null",
  "user": "null"
}
```

Returns:
```json
{
  "data": [
    {
      "embedding": [
        0
      ],
      "index": 0,
      "object": "embedding"
    }
  ],
  "model": "thenlper/gte-base",
  "object": "list",
  "usage": {
    "prompt_tokens": 512,
    "total_tokens": 512
  }
}
```
The OpenAPI.json is available under https://model-xxxxxx.api.baseten.co/environments/production/sync/openapi.json for more details.

#### Advanced:
You may also use Baseten's async jobs API, which returns a request_id, which you can use to query the status of the job and get the results.

POST-Route: `https://model-xxxxxx.api.baseten.co/environments/production/async/v1/embeddings`
Read more about [Baseten's Async API here](https://docs.baseten.co/invoke/async)

### curl
```bash
curl -X POST https://model-xxxxxx.api.baseten.co/environments/production/sync/v1/embeddings \
        -H "Authorization: Api-Key YOUR_API_KEY" \
        -d '{"input": "text string", "model": "model"}'
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
texts = ["Hello world", "Example text", "Another sample"]
response = client.embed(
    input=texts,
    model="my_model",
    batch_size=4,
    max_concurrent_requests=32,
    timeout_s=360
)
print(response.numpy())
```

Read more on the [Baseten Performance Client Blog](https://www.baseten.co/blog/your-client-code-matters-10x-higher-embedding-throughput-with-python-and-rust/)

### OpenAI compatible client library
```python
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.environ['BASETEN_API_KEY'],
    base_url="https://model-xxxxxx.api.baseten.co/environments/production/sync/v1"
)

embedding = client.embeddings.create(
    input="Baseten Embeddings are fast",
    model="model"
)
```
### requests python library

```python
import os
import requests

resp = requests.post(
    "https://model-xxxxxx.api.baseten.co/environments/production/sync/v1/embeddings",
    headers={"Authorization": "Api-Key " + str(os.environ['BASETEN_API_KEY'])},
    json={"input": ["text string", "second string"]},
)

print(resp.json())
```


## Config.yaml
By default, the following configuration is used for this deployment.

```yaml
base_image:
  image: baseten/text-embeddings-inference-mirror:89-1.7.2
docker_server:
  liveness_endpoint: /health
  predict_endpoint: /v1/embeddings
  readiness_endpoint: /health
  server_port: 7997
  start_command: bash -c "truss-transfer-cli && text-embeddings-router --port 7997
    --model-id /app/model_cache/cached_model --max-client-batch-size 128 --max-concurrent-requests
    1024 --max-batch-tokens 16384 --auto-truncate --tokenization-workers 3"
model_cache:
- ignore_patterns:
  - '*.pt'
  - '*.ckpt'
  - '*.onnx'
  repo_id: mixedbread-ai/mxbai-embed-large-v1
  revision: main
  use_volume: true
  volume_folder: cached_model
model_metadata:
  example_model_input:
    encoding_format: float
    input: text string
    model: model
model_name: TEI-mixedbread-ai-mxbai-embed-large-v1-embedding-truss-example
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
