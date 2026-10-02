# Rune 26B A4B typed decisions

Deploy [michaelfeil/rune-26b-a4b](https://huggingface.co/michaelfeil/rune-26b-a4b) using BEI-Bert's `encoder_bert` integration. The config pins the checkpoint revision and serving version.

## Deploy

From the repository root:

```bash
pip install --upgrade truss
truss push 11-embeddings-reranker-classification-tensorrt/BEI-Bert-michaelfeil-rune-26b-a4b --publish
```

The example uses BF16 and sets `DECISION_PROTOCOL=rune` to select the checkpoint’s trained decision protocol. Each question is a separate sequence in the inference batch; one request returns all answers. Use the explicit `/v1/systemone` route.
This is the Gemma4-based **26B A4B** checkpoint, rather than a 26B A3B checkpoint. The example uses text input and does not configure image fetching. H100 provides room for the full BF16 weights and inference workspace.

Rune rejects prompts that exceed its input limit; it never truncates decision prompts. Keep the state, instructions, and options within that limit.

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

Use `H100:2` for two independent model replicas when more throughput is needed. Each GPU must fit the complete model; adding GPUs does not split the weights or lower the memory required per replica. The BF16 26B checkpoint does not fit on an individual L4 or 48 GB RTX6000.
