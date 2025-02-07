# Baseten-Embeddings-Inference with SamLowe/roberta-base-go_emotions-classification

This is a Deployment for BEI (Baseten-Embeddings-Inference) with SamLowe/roberta-base-go_emotions-classification. BEI is Basetens soution for production-grade deployments via TensorRT-LLM. 

With BEI you get the following benefits:
- low-latency (sub 6ms latency) 
- high user queries: (up to 1400 requests per second)
- high-throughput inference - highest tokens / flops across any embedding solution (XQA kernels and dynamic batching)
- cached model weights for fast vertical scaling and high availability (No huggingface hub dependency at runtime)

This deployment is specifically designed for the huggingface model [SamLowe/roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions).
It might be also working for other models that have the architecture of RobertaForSequenceClassification specificied in their huggingface transformers config.
Suitable models can be identified by the `ForSequenceClassification` suffix in the model name. Prediction models may have one or more labels, which are returned with the prediction.

SamLowe/roberta-base-go_emotions  is a text-classification model, used to classify a text into a category. 
 It is frequently used in sentiment analysis, spam detection, and more. Its also used for deployment of chat rating models, e.g. RLHF reward models or toxicity detection models.


## Deployment with Truss

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`


First, clone this repository:
```sh
git clone https://github.com/basetenlabs/truss-examples.git
cd 11-embeddings-reranker-classification-tensorrt/BEI-samlowe-roberta-base-go_emotions-classification
```

With `11-embeddings-reranker-classification-tensorrt/BEI-samlowe-roberta-base-go_emotions-classification` as your working directory, you can deploy the model with the following command, paste your Baseten API key if prompted.

```sh
truss push --publish
# prints: 
# ✨ Model BEI-samlowe-roberta-base-go_emotions-classification-truss-example was successfully pushed ✨
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

```json
[
  {
    "label": "excitment",
    "score": 0.99
  }
]
```
Important, this is different from the `predict` route: `https://model-xxxxxx.api.baseten.co/environments/production/predict`
The OpenAPI.json is available under `https://model-xxxxxx.api.baseten.co/environments/production/sync/openapi.json` for more details.

#### Advanced:
You may also use Baseten's async jobs api, which returns a request_id, which you can use to query the status of the job and get the results.
POST-Route: https://model-xxxxxx.api.baseten.co/environments/production/sync

### OpenAI compatible client library
OpenAI does not have a classification endpoint, therefore no client library is available.


## Config.yaml
By default, the following configuration is used for this deployment. If you want to remove the quantization, remove the `quantization_type` field or set it to `no_quant` for float16.

```yaml
build_commands: []
environment_variables: {}
external_package_dirs: []
model_metadata:
  example_model_input:
    input: This redirects to the embedding enpoint. Use the /sync api to reach /sync/predict
      endpoint.
model_name: BEI-samlowe-roberta-base-go_emotions-classification-truss-example
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
      repo: SamLowe/roberta-base-go_emotions
      revision: main
      source: HF
    max_num_tokens: 16384
    max_seq_len: 1000001

```

## Support
If you have any questions or need assistance, please open an issue in this repository or contact our support team.
