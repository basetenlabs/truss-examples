# BEI (Baseten-Embeddings-Inference) with papluca/xlm-roberta-base-language-detection-classification

This is a Deployment for BEI (Baseten-Embeddings-Inference) with papluca/xlm-roberta-base-language-detection-classification. BEI is Baseten's solution for production-grade deployments via TensorRT-LLM for (text) embeddings, reranking models and prediction models.
With BEI you get the following benefits:
- *Lowest-latency inference* across any embedding solution (vLLM, SGlang, Infinity, TEI, Ollama)<sup>1</sup>
- *Highest-throughput inference* across any embedding solution (vLLM, SGlang, Infinity, TEI, Ollama) - thanks to XQA kernels, FP8 and dynamic batching.<sup>2</sup>
- High parallelism: up to 1400 client embeddings per second
- Cached model weights for fast vertical scaling and high availability - no Hugging Face hub dependency at runtime


# Examples:
This deployment is specifically designed for the Hugging Face model [papluca/xlm-roberta-base-language-detection](https://huggingface.co/papluca/xlm-roberta-base-language-detection).
Suitable models can be identified by the `ForSequenceClassification` suffix in the model name. Prediction models may have one or more labels, which are returned with the prediction.

papluca/xlm-roberta-base-language-detection  is a text-classification model, used to classify a text into a category. \nIt is frequently used in sentiment analysis, spam detection, and more. It's also used for deployment of chat rating models, e.g. RLHF reward models or toxicity detection models.


## Deployment with Truss

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`


First, clone this repository:
```sh
git clone https://github.com/basetenlabs/truss-examples.git
cd 11-embeddings-reranker-classification-tensorrt/BEI-papluca-xlm-roberta-base-language-detection-classification
```

With `11-embeddings-reranker-classification-tensorrt/BEI-papluca-xlm-roberta-base-language-detection-classification` as your working directory, you can deploy the model with the following command. Paste your Baseten API key if prompted.

```sh
truss push --publish
# prints:
# ✨ Model BEI-papluca-xlm-roberta-base-language-detection-classification-truss-example was successfully pushed ✨
# 🪵  View logs for your deployment at https://app.baseten.co/models/yyyyyy/logs/xxxxxx
```

## Call your model

### API-Schema:
POST-Route: `https://model-xxxxxx.api.baseten.co/environments/production/sync/predict`
```json
{
  "inputs": "Baseten is a fast inference provider",
  "raw_scores": false,
  "truncate": false,
  "truncation_direction": "right"
}
```

Returns:
```json
[
  {
    "label": "excitement",
    "score": 0.99
  }
]
```
Important, this is different from the `predict` route: https://model-xxxxxx.api.baseten.co/environments/production/predict
The OpenAPI.json is available under https://model-xxxxxx.api.baseten.co/environments/production/sync/openapi.json for more details.

#### Advanced:
You may also use Baseten's async jobs API, which returns a request_id, which you can use to query the status of the job and get the results.

POST-Route: `https://model-xxxxxx.api.baseten.co/environments/production/async/predict`
Read more about [Baseten's Async API here](https://docs.baseten.co/invoke/async)

### OpenAI compatible client library
OpenAI does not have a classification endpoint, therefore no client library is available.


## Config.yaml
By default, the following configuration is used for this deployment.

```yaml
build_commands: []
environment_variables: {}
external_package_dirs: []
model_metadata:
  example_model_input:
    input: This redirects to the embedding endpoint. Use the /sync API to reach /sync/predict
      endpoint.
model_name: BEI-papluca-xlm-roberta-base-language-detection-classification-truss-example
python_version: py39
requirements: []
resources:
  accelerator: L4
  cpu: '1'
  memory: 10Gi
  use_gpu: true
secrets: {}
system_packages: []
trt_llm:
  build:
    base_model: encoder
    checkpoint_repository:
      repo: papluca/xlm-roberta-base-language-detection
      revision: main
      source: HF
    max_num_tokens: 16384
    max_seq_len: 1000001

```

## Support
If you have any questions or need assistance, please open an issue in this repository or contact our support team.
