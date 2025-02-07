# Baseten-Embeddings-Inference with BAAI/bge-reranker-large

This is a Deployment for BEI (Baseten-Embeddings-Inference) with BAAI/bge-reranker-large. BEI is Basetens soution for production-grade deployments via TensorRT-LLM. 

With BEI you get the following benefits:
- low-latency (sub 6ms latency) 
- high user queries: (up to 1400 requests per second)
- high-throughput inference - highest tokens / flops across any embedding solution (XQA kernels and dynamic batching)
- cached model weights for fast vertical scaling and high availability (No huggingface hub dependency at runtime)

This deployment is specifically designed for the huggingface model [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large).
It might be also working for other models that have the architecture of XLMRobertaForSequenceClassification specificied in their huggingface transformers config.
Suitable models can be identified by the `ForSequenceClassification` suffix in the model name. Reranker models may have AT MOST ONE labels, which contains the score of the reranking.

BAAI/bge-reranker-large  is a reranker model, used to re-rank a list of items, given a query. 
 It is frequently used in search engines, recommendation systems, and more.


## Deployment with Truss

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`


First, clone this repository:
```sh
git clone https://github.com/basetenlabs/truss-examples.git
cd 11-embeddings-reranker-classification-tensorrt/BEI-baai-bge-reranker-large
```

With `11-embeddings-reranker-classification-tensorrt/BEI-baai-bge-reranker-large` as your working directory, you can deploy the model with the following command, paste your Baseten API key if prompted.

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
  "raw_scores": false,
  "return_text": false,
  "texts": [
    "Deep Learning is ...", "Baseten is a fast inference provider"
  ],
  "truncate": false,
  "truncation_direction": "right"
}

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
The OpenAPI.json is available under `https://model-xxxxxx.api.baseten.co/environments/production/sync/openapi.json` for more details.

#### Advanced:
You may also use Baseten's async jobs api, which returns a request_id, which you can use to query the status of the job and get the results.
POST-Route: https://model-xxxxxx.api.baseten.co/environments/production/sync/rerank`
Read more about the [Baseten's Async API here ](https://docs.baseten.co/invoke/async)


### OpenAI compatible client library
OpenAI.com does not have a rerank endpoint, therefore no client library is available.



## Config.yaml
By default, the following configuration is used for this deployment. If you want to remove the quantization, remove the `quantization_type` field or set it to `no_quant` for float16.

```yaml
build_commands: []
environment_variables: {}
external_package_dirs: []
model_metadata:
  example_model_input:
    input: This redirects to the embedding enpoint. Use the /sync api to reach /rerank
model_name: BEI-baai-bge-reranker-large-truss-example
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
      repo: BAAI/bge-reranker-large
      revision: main
      source: HF
    max_num_tokens: 16384
    max_seq_len: 1000001

```

## Support
If you have any questions or need assistance, please open an issue in this repository or contact our support team.
