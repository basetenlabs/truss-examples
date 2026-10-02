# BEI-Bert (Baseten-Embeddings-Inference-BERT) with Cloudflare/clef

This is a Deployment for BEI-Bert (Baseten-Embeddings-Inference-BERT) with Cloudflare/clef. BEI is Baseten's solution for production-grade deployments via TensorRT-LLM for (text) embeddings, reranking models and prediction models.
With BEI you get the following benefits:
- *Lowest-latency inference* across any embedding solution (vLLM, SGlang, Infinity, TEI, Ollama)<sup>1</sup>
- *Highest-throughput inference* across any embedding solution (vLLM, SGlang, Infinity, TEI, Ollama) - thanks to XQA kernels, FP8 and dynamic batching.<sup>2</sup>
- High parallelism: up to 1400 client embeddings per second
- Cached model weights for fast vertical scaling and high availability - no Hugging Face hub dependency at runtime


# Examples:
This deployment is specifically designed for the Hugging Face model [Cloudflare/clef](https://huggingface.co/Cloudflare/clef).
Requires a checkpoint with a trained typed-decision head or protocol.

Cloudflare/clef Answers choice, boolean, and score questions about a shared state.


Set `DECISION_PROTOCOL=clef` to load the joint schema head. All questions in a request share one backbone pass. This deployment supports text state and text messages; image and video inputs are rejected. Prompts exceeding the configured token budget return 422 instead of being truncated.

### Multiple GPUs

Use `H100:2` for two independent model replicas. Each GPU must fit the complete model; the BF16 27B checkpoint does not fit on an individual L4 or 48 GB RTX6000.

## Deployment with Truss

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`


First, clone this repository:
```sh
git clone https://github.com/basetenlabs/truss-examples.git
cd truss-examples/11-embeddings-reranker-classification-tensorrt/BEI-Bert-cloudflare-clef
```

With `11-embeddings-reranker-classification-tensorrt/BEI-Bert-cloudflare-clef` as your working directory, you can deploy the model with the following command. Paste your Baseten API key if prompted.

```sh
truss push --publish
# prints:
# ✨ Model BEI-Bert-cloudflare-clef-truss-example was successfully pushed ✨
# 🪵  View logs for your deployment at https://app.baseten.co/models/yyyyyy/logs/xxxxxx
```

## Call your model

Call the explicit `/v1/systemone` route. The response contains an answer for each question.

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
  DECISION_PROTOCOL: clef
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
model_name: BEI-Bert-cloudflare-clef-truss-example
python_version: py313
resources:
  accelerator: H100
  cpu: '1'
  memory: 80Gi
  use_gpu: true
trt_llm:
  build:
    base_model: encoder_bert
    checkpoint_repository:
      repo: Cloudflare/clef
      revision: 2f3de3dd85f379784083b0814d997ab627200f0c
      source: HF
    max_num_tokens: 8192
    pipeline_parallel_count: 1
    sequence_parallel_count: 1
    tensor_parallel_count: 1
  runtime:
    webserver_default_route: null

```

## Support
If you have any questions or need assistance, please open an issue in this repository or contact our support team.
