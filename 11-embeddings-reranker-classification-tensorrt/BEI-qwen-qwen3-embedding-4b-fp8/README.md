# BEI (Baseten-Embeddings-Inference) with Qwen/Qwen3-Embedding-4B

This is a Deployment for BEI (Baseten-Embeddings-Inference) with Qwen/Qwen3-Embedding-4B. BEI is Baseten's solution for production-grade deployments via TensorRT-LLM for (text) embeddings, reranking models and prediction models.
With BEI you get the following benefits:
- *Lowest-latency inference* across any embedding solution (vLLM, SGlang, Infinity, TEI, Ollama)<sup>1</sup>
- *Highest-throughput inference* across any embedding solution (vLLM, SGlang, Infinity, TEI, Ollama) - thanks to XQA kernels, FP8 and dynamic batching.<sup>2</sup>
- High parallelism: up to 1400 client embeddings per second
- Cached model weights for fast vertical scaling and high availability - no Hugging Face hub dependency at runtime


# Examples:
This deployment is specifically designed for the Hugging Face model [michaelfeil/Qwen3-Embedding-4B-auto](https://huggingface.co/michaelfeil/Qwen3-Embedding-4B-auto).
Suitable models need to have the configurations of the `sentence-transformers` library, which are used for embeddings. Such repos contain e.g. a `sbert_config.json` or a `1_Pooling/config.json` file besides the fast-tokenizer and the safetensors file.

michaelfeil/Qwen3-Embedding-4B-auto  is a text-embeddings model, producing a 1D embeddings vector, given an input.
It's frequently used for downstream tasks like clustering, used with vector databases.

This model is quantized to FP8 for deployment, which is supported by Nvidia's newest GPUs e.g. H100, H100_40GB or L4. Quantization is optional, but leads to higher efficiency.

## Deployment with Truss

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`


First, clone this repository:
```sh
git clone https://github.com/basetenlabs/truss-examples.git
cd 11-embeddings-reranker-classification-tensorrt/BEI-qwen-qwen3-embedding-4b-fp8
```

With `11-embeddings-reranker-classification-tensorrt/BEI-qwen-qwen3-embedding-4b-fp8` as your working directory, you can deploy the model with the following command. Paste your Baseten API key if prompted.

```sh
truss push --publish
# prints:
# ✨ Model BEI-qwen-qwen3-embedding-4b-fp8-truss-example was successfully pushed ✨
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
By default, the following configuration is used for this deployment. This config uses `quantization_type=fp8`. This is optional, remove the `quantization_type` field or set it to `no_quant` for float16/bfloat16.

```yaml
model_metadata:
  example_model_input:
    encoding_format: float
    input: text string
    model: model
model_name: BEI-qwen-qwen3-embedding-4b-fp8-truss-example
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
      repo: michaelfeil/Qwen3-Embedding-4B-auto
      revision: main
      source: HF
    max_num_tokens: 40960
    num_builder_gpus: 1
    quantization_type: fp8
  runtime:
    webserver_default_route: /v1/embeddings
  version_overrides:
    bei_version: 0.0.25-b200-dev-v4
    engine_builder_version: 0.20.0.dev1

```

## Support
If you have any questions or need assistance, please open an issue in this repository or contact our support team.
