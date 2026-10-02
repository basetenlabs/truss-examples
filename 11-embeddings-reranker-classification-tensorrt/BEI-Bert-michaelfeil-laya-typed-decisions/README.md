# BEI-Bert (Baseten-Embeddings-Inference-BERT) with michaelfeil/laya-typed-decisions

This is a Deployment for BEI-Bert (Baseten-Embeddings-Inference-BERT) with michaelfeil/laya-typed-decisions. BEI is Baseten's solution for production-grade deployments via TensorRT-LLM for (text) embeddings, reranking models and prediction models.
With BEI you get the following benefits:
- *Lowest-latency inference* across any embedding solution (vLLM, SGlang, Infinity, TEI, Ollama)<sup>1</sup>
- *Highest-throughput inference* across any embedding solution (vLLM, SGlang, Infinity, TEI, Ollama) - thanks to XQA kernels, FP8 and dynamic batching.<sup>2</sup>
- High parallelism: up to 1400 client embeddings per second
- Cached model weights for fast vertical scaling and high availability - no Hugging Face hub dependency at runtime


# Examples:
This deployment is specifically designed for the Hugging Face model [michaelfeil/laya-typed-decisions](https://huggingface.co/michaelfeil/laya-typed-decisions).
Requires a checkpoint with a trained typed-decision head or protocol.

michaelfeil/laya-typed-decisions Answers choice, boolean, and score questions about a shared state.


Laya accepts plain-text state. Serialize structured state to text. BF16 is recommended; each question is formatted and truncated according to checkpoint limits.

### Multiple GPUs

For independent replicas in one deployment, change `resources.accelerator` to `L4:2` or `L4:4`. Each GPU holds the whole model; weights are not split between GPUs. Start with one L4, then measure throughput and queueing at your target load before adding replicas. RTX6000 deployments follow the same full-model-per-GPU capacity requirement where that accelerator is available.

## Deployment with Truss

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`


First, clone this repository:
```sh
git clone https://github.com/basetenlabs/truss-examples.git
cd truss-examples/11-embeddings-reranker-classification-tensorrt/BEI-Bert-michaelfeil-laya-typed-decisions
```

With `11-embeddings-reranker-classification-tensorrt/BEI-Bert-michaelfeil-laya-typed-decisions` as your working directory, you can deploy the model with the following command. Paste your Baseten API key if prompted.

```sh
truss push --publish
# prints:
# ✨ Model BEI-Bert-michaelfeil-laya-typed-decisions-truss-example was successfully pushed ✨
# 🪵  View logs for your deployment at https://app.baseten.co/models/yyyyyy/logs/xxxxxx
```

## Call your model

Call the explicit `/v1/systemone` route. Each question is a separate inference sequence; the response contains all answers.

```bash
export MODEL_URL="https://model-${MODEL_ID}.api.baseten.co/environments/production/sync"
curl --fail-with-body "$MODEL_URL/v1/systemone" \
  -H "Authorization: Api-Key $BASETEN_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
  "state": "I was billed twice. Please refund the extra charge.",
  "questions": {
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this request?",
      "criteria": {
        "billing": "Billing and refunds",
        "technical_support": "Technical troubleshooting"
      }
    },
    "refund": {
      "type": "noul",
      "instructions": "Is the customer asking for a refund?"
    }
  }
}'
```


## Config.yaml
By default, the following configuration is used for this deployment.

```yaml
environment_variables:
  AUTO_TRUNCATE: 'true'
  DTYPE: bfloat16
model_metadata:
  example_model_input:
    questions:
      refund:
        instructions: Is the customer asking for a refund?
        type: noul
      team:
        criteria:
          billing: Billing and refunds
          technical_support: Technical troubleshooting
        instructions: Which team should handle this request?
        type: choice
    state: I was billed twice. Please refund the extra charge.
model_name: BEI-Bert-michaelfeil-laya-typed-decisions-truss-example
python_version: py313
resources:
  accelerator: L4
  cpu: '1'
  memory: 10Gi
  use_gpu: true
trt_llm:
  build:
    base_model: encoder_bert
    checkpoint_repository:
      repo: michaelfeil/laya-typed-decisions
      revision: 0b6de4ff4ee8b16c83011c7f107a39ca191008f5
      source: HF
    max_num_tokens: 8192
    pipeline_parallel_count: 1
    sequence_parallel_count: 1
    tensor_parallel_count: 1
  runtime:
    webserver_default_route: null
  version_overrides:
    bei_bert_version: 1.8.16

```

## Support
If you have any questions or need assistance, please open an issue in this repository or contact our support team.
