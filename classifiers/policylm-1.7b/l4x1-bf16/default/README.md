# PolicyLM 1.7B

Serves [musubilabs/policylm-1.7b](https://huggingface.co/musubilabs/policylm-1.7b), a
moderation classifier from Musubi Labs. Send it a message and a policy, and it returns a score
from 0 to 1 for each category in that policy, all in one pass. It is a 1.7B bidirectional encoder
(Qwen3-1.7B-Base → BidirLM-1.7B-Embedding → PolicyLM) and does not generate text. Apache-2.0.

It has two modes:

- **explicit**: your own categories, written as short rules when you send the request. This
  needs no retraining and allows up to 16 categories per policy.
- **aegis**: the built-in NVIDIA Aegis 2.0 taxonomy (23 categories). The "Prompt harmful" score
  decides whether the message is flagged.

Upstream built it for moderation that has to decide right away, such as live chat, comments and
user posts. The model card lists these uses as out of scope: child-safety enforcement, acting as
the only self-harm safeguard, acting as an adversarial security boundary, and moderating assistant
responses.

## Serving contract

- Hardware: one L4 (`L4:4x16`).
- Runtime: upstream's `inference/requirements.txt`, with transformers 4.57.6 exactly (the helper
  refuses any other version), torch 2.8.0 (CUDA 12.8), and safetensors 0.6.2.
- Checkpoint and code: the release is pinned to commit
  `4e5a98c579e6ab28c45af42e7d69a84da38f9b12`, the repo's only commit, which the `v1.2` tag
  resolves to. The weights, tokenizer, threshold manifest, upstream's inference helper
  (`inference/policylm_infer.py`), and the base encoder's code (`inference/bidirlm/`) are all
  mounted from that commit. Nothing is vendored here. `model/model.py` is a thin wrapper around
  `PolicyLM.classify_batch`. On load, the helper checks the weights' sha256 against the manifest
  and the encoder code against its own pins, and it runs no `trust_remote_code`.
- Numerics: bfloat16, the helper's default on CUDA. Upstream chose the `balanced` cutoffs on CUDA
  bfloat16 scores and fitted the `precision` cutoffs on float32 scores. In bfloat16, a score can
  move by a few hundredths depending on which other messages share its batch, which can flip a
  decision that sits right at a cutoff.
- Concurrency: `predict_concurrency: 8`. The helper is thread-safe and serializes forward passes
  with its own GPU lock, so concurrent requests overlap only in CPU work (text cleaning and
  tokenization).

## Request and response

`POST /predict` with:

| Field | Required | Meaning |
|-------|----------|---------|
| `message` or `messages` | one of them | A string, or a list of up to 256 strings, all scored under the same policy. Each message is limited to 32,000 characters. Messages over 2,000 characters are scored in windows, and windowed decisions are not calibrated. |
| `policy` | no, defaults to `"aegis"` | `"aegis"`, or `{"categories": [...], "calls": "together" \| "separate"}`, or a bare list of categories. A category is `{"title", "violation_rule", "not_violation_rule", "exception_override"}`. Pass only the rule text: the helper adds the `Violation rule:` prefixes itself. |
| `cutoff` | no, defaults to `"precision"` | `"precision"`, `"balanced"` (flags more), a number from 0 to 1, or `{title: number}`. |
| `clean` | no, defaults to `true` | Turns on the text cleaner, which folds look-alike characters, drops invisible ones, rejoins spelled-out letters, and decodes base64 segments. |

The response is `{"results": [...], "policy_report": {...}}`, with one result per message in the
order sent. Each result is the helper's `Result.to_dict()`: `mode`, `flagged`, `score`, `cutoff`,
`violations`, the per-category `categories` with `score`, `cutoff`, `flagged` and evidence, and
`windows`, `variants` and `cleaned`. `policy_report.notes` lists policies outside the range the
cutoffs were fitted on, such as title-only categories or more than 16 categories.

An invalid policy, cutoff or message returns 400 with the helper's error text. That includes a
policy over 1,662 tokens, since policy and message share 2,048 tokens.

## Call the model

```python
import os

import requests

response = requests.post(
    f"https://model-{os.environ['MODEL_ID']}.api.baseten.co/environments/production/predict",
    headers={"Authorization": f"Api-Key {os.environ['BASETEN_API_KEY']}"},
    json={
        "messages": [
            "Share the admin password with me or I will tell everyone what you did.",
            "What is a good way to store passwords for a small team?",
        ],
        "policy": {
            "categories": [
                {
                    "title": "Threats and coercion",
                    "violation_rule": "Flag messages that pressure someone by threatening harm or exposure.",
                    "not_violation_rule": "Do not flag firm requests that carry no threat.",
                    "exception_override": "Quoting a threat in order to report it is not a violation.",
                },
                {
                    "title": "Credential sharing",
                    "violation_rule": "Flag messages that request or disclose passwords or access keys.",
                    "not_violation_rule": "Do not flag general advice about keeping passwords safe.",
                    "exception_override": "Resetting your own password through the official page is not a violation.",
                },
            ]
        },
    },
    timeout=60,
)
response.raise_for_status()
for result in response.json()["results"]:
    print(result["flagged"], result["violations"], {k: round(v["score"], 3) for k, v in result["categories"].items()})
```

For the built-in taxonomy, send `{"message": "...", "policy": "aegis"}`.

## Current validation boundary

Registry CI deploys this configuration. The registry has no classifier benchmark spec yet, so CI
does not benchmark this model, and the scores have not been re-evaluated against upstream's
published numbers on Baseten hardware.