# Baseten-Embeddings-Inference with BAAI/bge-large-en-v1.5-embedding

This is a Deployment for BEI (Baseten-Embeddings-Inference) with BAAI/bge-large-en-v1.5-embedding. BEI is Basetens soution for production-grade deployments via TensorRT-LLM. 

With BEI you get the following benefits:
- low-latency (sub 6ms latency) 
- high user queries: (up to 1400 requests per second)
- high-throughput inference - highest tokens / flops across any embedding solution (XQA kernels and dynamic batching)
- cached model weights for fast vertical scaling and high availability (No huggingface hub dependency at runtime)

This deployment is specifically designed for the huggingface model [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5).
It might be also working for other models that have the architecture of BertModel specificied in their huggingface transformers config.
Suitable models need to have the configurations of the `sentence-transformers` library, which are used for embeddings. Such Repos contain e.g. a `sbert_config.json` or a `1_Pooling/config.json` file besides the fast-tokenizer and the safetensors file.

BAAI/bge-large-en-v1.5  is a text-embeddings model, producing a 1D-embeddings vector, given an output. 
 Its frequently used for downstream tasks like clustering, used with vector-databases.


## Deployment with Truss

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`


First, clone this repository:
```sh
git clone https://github.com/basetenlabs/truss-examples.git
cd 11-embeddings-reranker-classification-tensorrt/BEI-baai-bge-large-en-v1.5-embedding
```

With `11-embeddings-reranker-classification-tensorrt/BEI-baai-bge-large-en-v1.5-embedding` as your working directory, you can deploy the model with the following command, paste your Baseten API key if prompted.

```sh
truss push --publish
# prints: 
# ✨ Model BEI-baai-bge-large-en-v1.5-embedding-truss-example was successfully pushed ✨
# 🪵  View logs for your deployment at https://app.baseten.co/models/yyyyyy/logs/xxxxxx
```

## Call your model

### API-Schema:
POST-Route: https://model-xxxxxx.api.baseten.co/environments/production/sync/v1/embeddings`
```json
{
  "encoding_format": "float", # or base64
  "input": "string", # can be list of strings for multiple embeddings
  "model": "null", 
  "user": "null"
}

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
Advanced:
You may also use Baseten's async jobs api, which returns a request_id, which you can use to query the status of the job and get the results.
POST-Route: https://model-xxxxxx.api.baseten.co/environments/production/async/v1/embeddings`
Read more about the [Baseten's Async API here ](https://docs.baseten.co/invoke/async)

### curl
```bash
curl -X POST https://model-xxxxxx.api.baseten.co/environments/production/sync/v1/embeddings         -H "Authorization: Api-Key YOUR_API_KEY"         -d '{"input": "text string"}'
```

### OpenAI compatible client library
```python
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.environ['BASETEN_API_KEY'], 
    api_url="https://model-xxxxxx.api.baseten.co/environments/production/sync"
)

embedding = client.embeddings.create(
    input="Baseten Embeddings are fast",
    model="model"
)
```
### request python library

```python
import os
import requests

resp = requests.post(
    "https://model-xxxxxx.api.baseten.co/environments/production/sync/v1/embeddings",
    headers={"Authorization": "Api-Key "+str(os.environ['BASETEN_API_KEY'])},
    json={"input": ["text string", "second string"]},
)

print(resp.json())
```


## Config.yaml
By default, the following configuration is used for this deployment. If you want to remove the quantization, remove the `quantization_type` field or set it to `no_quant` for float16.

```yaml
build_commands: []
environment_variables: {}
external_package_dirs: []
model_metadata:
  example_model_input:
    encoding_format: float
    input: text string
    model: model
model_name: BEI-baai-bge-large-en-v1.5-embedding-truss-example
python_version: py39
requirements: []
resources:
  accelerator: L4
  cpu: '1'
  memory: 2Gi
  use_gpu: true
secrets: {}
system_packages: []
trt_llm:
  build:
    base_model: encoder
    checkpoint_repository:
      repo: BAAI/bge-large-en-v1.5
      revision: main
      source: HF
    max_num_tokens: 16384
    max_seq_len: 1000001

```

## Support
If you have any questions or need assistance, please open an issue in this repository or contact our support team.
