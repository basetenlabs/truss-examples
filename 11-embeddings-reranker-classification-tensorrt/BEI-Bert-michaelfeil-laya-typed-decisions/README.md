# Laya typed decisions

Deploy [michaelfeil/laya-typed-decisions](https://huggingface.co/michaelfeil/laya-typed-decisions) using BEI-Bert's `encoder_bert` integration. The config pins the checkpoint revision and serving version.

## Deploy

From the repository root:

```bash
pip install --upgrade truss
truss push 11-embeddings-reranker-classification-tensorrt/BEI-Bert-michaelfeil-laya-typed-decisions --publish
```

The example uses BF16 and the trained decision head. Each question is a separate sequence in the inference batch; one request returns all answers. Use the explicit `/v1/systemone` route.
Laya accepts plain-text state. Serialize structured state to text before sending it. BF16 is the default here: FP8 can change individual decisions and has not shown a consistent latency benefit for this model.

Laya formats and truncates each question according to its checkpoint limits. Keep important state information within the input limit; `max_num_tokens` controls the batch budget.

## Invoke

Set `MODEL_ID` to your deployment's model ID and `BASETEN_API_KEY` to your API key:

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

## Multiple GPUs

For independent replicas in one deployment, change `resources.accelerator` to `L4:2` or `L4:4`. Each GPU holds the whole model; weights are not split between GPUs. Start with one L4, then measure throughput and queueing at your target load before adding replicas. RTX6000 deployments follow the same full-model-per-GPU capacity requirement where that accelerator is available.
