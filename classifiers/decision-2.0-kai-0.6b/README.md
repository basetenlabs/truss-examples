# Decision 2.0 Kai 0.6B

Deploy [vLLM Semantic Router's Kai 0.6B](https://huggingface.co/vllm-sr/Decision-2.0-Kai-0.6B)
as a SystemOne HTTP server on one NVIDIA L4. Kai evaluates candidates and returns
decision probabilities. It does not generate text or require a chat-completions API.

The example uses the upstream Transformers-compatible Python implementation,
not the separate vLLM Semantic Router server. `model/model.py` loads the model in
`load()` and calls `system_one()` from `predict()`.

## Deploy

From the repository root, authenticate with your intended Baseten workspace and run:

```bash
uvx truss login
uvx truss push classifiers/decision-2.0-kai-0.6b
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

Example response, illustrating the response shape (probabilities and token counts vary):

```json
{
  "model": "Decision-2.0-Kai-0.6B",
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
[model card](https://huggingface.co/vllm-sr/Decision-2.0-Kai-0.6B#quickstart) for the full
contract. The default configuration serves both `/predict` and `/v1/systemone`.

## SystemOne API

The custom server reuses the same model loader and adds `POST /v1/systemone` and
`GET /v1/models`. `/predict` continues to accept the payload above.

After deployment, send the same payload with `"model": "Decision-2.0-Kai-0.6B"`
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
    model="Decision-2.0-Kai-0.6B",
    base_url=f"https://model-{model_id}.api.baseten.co/environments/production/sync",
    http_client=httpx2.Client(event_hooks={"request": [baseten_auth]}, timeout=120),
) as client:
    result = client.system_one(
        "Please send a replacement today.",
        {"urgent": Noul(instructions="Does the customer request action today?")},
    )
    print(result.nouls["urgent"].noul)
```

Kai supports up to 8,192 tokens per question, 2–255 Choice candidates,
2–10 Score levels, and the adapter permits at most 1,024 questions per request. Missing or null
instructions become an empty string. Invalid requests and upstream input errors
return HTTP 422; invalid model outputs return HTTP 500. Inference is serialized
by the adapter, including requests through `/sync`.

## Runtime and limits

- Weights and inference code are pinned together to Hugging Face revision
  `cd49ea3813fd8ba0928a9a23ef6c9a0f2f0cd764`. Dependencies are version-pinned,
  including Transformers 5.17.0.
- Every manifest-listed checkpoint file is mounted, including the nested `backbone/`
  weights, `decision2/` runtime, tokenizer, score calibration, and manifest.
  Upstream verifies the exact file inventory and SHA-256 hashes, including the
  README, license, and assets; a weights-only filter would fail verification.
- The loader uses upstream `AutoModel.from_pretrained()` with
  `trust_remote_code=True` and `local_files_only=True`. The pinned wrapper loads
  the nested runtime. The adapter creates a temporary bundle containing only
  manifest-listed files, using hard links where possible and copies otherwise,
  so mount metadata and symlinks do not invalidate the inventory. Upstream still
  verifies every hash and parameter count. Checkpoint Python code executes
  during model loading.
- Each question, including state, instructions, candidates, and special tokens,
  must fit within 8,192 tokens. Upstream rejects oversized requests rather than
  truncating them. The HTTP adapter converts upstream question errors into HTTP
  422 and invalid model outputs into HTTP 500.
- GPU inference uses upstream BF16 autocast, with BF16-exact linear weights held
  in BF16 and the decision head and remaining tensors in FP32. The example keeps
  upstream's default exact path; shared-context inference is disabled.
- `predict_concurrency: 1` serializes requests. Multiple questions use the
  upstream batching implementation; concurrent HTTP requests can queue.

## Validation

CPU-only adapter tests cover the three decision types, model discovery, SDK
compatibility, error handling, and serialized inference:

```bash
PYTHONPATH=classifiers/decision-2.0-kai-0.6b uv run --python 3.11 \
  --with-requirements classifiers/decision-2.0-kai-0.6b/tests/requirements.txt \
  pytest classifiers/decision-2.0-kai-0.6b/tests
```

GPU deployment and prediction validation for this Decision 2.0 checkpoint are
tracked in PR [#664](https://github.com/basetenlabs/model-registry/pull/664).
The prior Decision 1.0 deployment results do not validate this checkpoint.

`/health` becomes available after loading and warm-up. `/metrics` exposes the
Python process metrics for the custom server.

Registry CI runs the [classifier JevBench profiles](https://github.com/basetenlabs/model-registry/blob/main/benchmarking/classifiers/README.md)
against `/v1/systemone`, reporting decision accuracy and invalid/missing decisions.