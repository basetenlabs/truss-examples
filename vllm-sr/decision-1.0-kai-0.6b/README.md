# Decision 1.0 Kai 0.6B

Deploy [vLLM Semantic Router's Kai 0.6B](https://huggingface.co/vllm-sr/Decision-1.0-Kai-0.6B)
as a standard Python Truss on one NVIDIA L4. Kai evaluates candidates and returns
decision probabilities. It does not generate text or require a chat-completions API.

The example uses the upstream Transformers-compatible Python implementation,
not the separate vLLM Semantic Router server. `model/model.py` loads the model in
`load()` and calls `system_one()` from `predict()`.

## Deploy

From the repository root, authenticate with your intended Baseten workspace and run:

```bash
uvx truss login
uvx truss push vllm-sr/decision-1.0-kai-0.6b
```

If you use named remotes, pass the same `--remote <remote-name>` to both commands.
The checkpoint is public; no Hugging Face access-token secret is required.
Wait for the deployment to become active before sending requests.

## Call

Set `BASETEN_API_KEY` to a key for the deployment's workspace and `MODEL_ID` to
the ID printed by the push command.

```bash
curl --fail-with-body -sS \
  "https://model-${MODEL_ID}.api.baseten.co/environments/production/predict" \
  -H "Authorization: Api-Key ${BASETEN_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "state": "The parcel arrived damaged. Please send a replacement today.",
    "questions": {
      "route": {
        "type": "choice",
        "instructions": "Which team should handle this request?",
        "criteria": {
          "delivery": "Damaged or missing parcels",
          "billing": "Payments and invoices"
        }
      },
      "urgent": {
        "type": "noul",
        "instructions": "Does the customer request action today?"
      }
    }
  }'
```

Example response, with probabilities rounded:

```json
{
  "model": "Decision-1.0-Kai-0.6B",
  "answers": {
    "route": {
      "type": "choice",
      "choice": "delivery",
      "confidence": 0.8808,
      "probabilities": {"delivery": 0.9404, "billing": 0.0596}
    },
    "urgent": {"type": "noul", "noul": 0.9588}
  },
  "usage": {"input_tokens": 87, "output_tokens": 0}
}
```

The payload preserves the upstream `state` and `questions` fields. Questions are
keyed by caller-supplied IDs. `choice` selects among candidates, `noul` returns a
probability that a condition holds, and `score` evaluates ordered criteria. See the
[model card](https://huggingface.co/vllm-sr/Decision-1.0-Kai-0.6B#use) for the full
contract. The default configuration exposes `/predict`. For a SystemOne HTTP
endpoint, use the optional configuration below.

## Optional SystemOne API

The custom server reuses the same model loader and adds `POST /v1/systemone` and
`GET /v1/models`. `/predict` continues to accept the payload above.

```bash
uvx truss push vllm-sr/decision-1.0-kai-0.6b \
  --config vllm-sr/decision-1.0-kai-0.6b/config-systemone.yaml
```

After deployment, send the same payload with `"model": "Decision-1.0-Kai-0.6B"`
to `https://model-${MODEL_ID}.api.baseten.co/environments/production/sync/v1/systemone`.
Keep the Baseten `Authorization: Api-Key ...` header. Model discovery is available
at the same prefix followed by `/v1/models`.

The [TypeSafe Python SDK](https://docs.typesafe.ai/sdk/python/usage) can call it
with a custom base URL and an HTTP hook for Baseten authentication:

```python
import os

import httpx2
from typesafe_sdk import Noul, TypeSafeClient

api_key = os.environ["BASETEN_API_KEY"]
model_id = os.environ["MODEL_ID"]


def baseten_auth(request):
    request.headers["Authorization"] = f"Api-Key {api_key}"


with TypeSafeClient(
    api_key=api_key,
    model="Decision-1.0-Kai-0.6B",
    base_url=f"https://model-{model_id}.api.baseten.co/environments/production/sync",
    http_client=httpx2.Client(event_hooks={"request": [baseten_auth]}, timeout=120),
) as client:
    result = client.system_one(
        "Please send a replacement today.",
        {"urgent": Noul(instructions="Does the customer request action today?")},
    )
    print(result.nouls["urgent"].noul)
```

Kai retains its 1,024-token limit per question, 2–255 Choice candidates,
2–10 Score levels, and at most 1,024 questions per request. Missing or null
instructions become an empty string. Invalid requests and upstream input errors
return HTTP 422; invalid model outputs return HTTP 500. Inference is serialized
by the adapter, including requests through `/sync`.

## Runtime and limits

- Weights and inference code are pinned together to Hugging Face revision
  `79263ba4c4befac3845e7c1111c2c679d0716623`. Dependencies are version-pinned.
- The weights manifest explicitly includes the nested encoder and tokenizer files.
  A `native/*` filter alone did not include those files in deployment testing.
- The loader imports the mounted `kai` package directly. With Transformers 4.57.6,
  `AutoModel.from_pretrained()` on the local mount missed the transitive
  `decision1_rocm_conv.py` helper in its dynamic-module cache. Direct imports keep
  relative imports in the complete pinned package. The source comes from the
  checkpoint repository and executes as part of model loading.
- Each question, including state, instructions, candidates, and special tokens,
  must fit within 1,024 tokens. Upstream rejects oversized requests rather than
  truncating them. Invalid questions may return per-question errors in the JSON
  response; HTTP 200 alone does not imply every question succeeded.
- Upstream fixes encoder numerics to FP32 and disables TF32 and fused attention
  fast paths. This example preserves those settings; it does not quantize weights.
- `predict_concurrency: 1` serializes requests. Multiple questions use the
  upstream batching implementation; concurrent HTTP requests can queue.
