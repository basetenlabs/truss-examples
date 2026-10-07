"""PolicyLM-1.7B moderation classifier, served through upstream's own inference helper.

The release (weights, tokenizer, threshold manifest) and its helper (inference/policylm_infer.py,
plus the vendored base-encoder code in inference/bidirlm/) are mounted together from the pinned
Hugging Face commit. Nothing is vendored here: the helper verifies the weights' sha256 against the
manifest and the encoder code against its own pins before it builds the model.

Request:
  {"messages": ["...", ...] | "message": "...",
   "policy": "aegis" | {"categories": [...], "calls": "together"|"separate"} | [category, ...],
   "cutoff": "precision"|"balanced"|<0-1>|{title: <0-1>},   # optional, default "precision"
   "clean": true|false}                                      # optional, default true
  A category is {"title", "violation_rule", "not_violation_rule", "exception_override"} or
  {"text": "<already rendered category>"} (the helper's Category.from_dict shape).

Response:
  {"results": [Result.to_dict(), ...],   # one per message, in order
   "policy_report": {"tokens", "content_tokens", "calls", "notes"}}

Invalid policies, cutoffs or messages (and messages over max_chars) return 400.
"""

import logging
import os
import sys

from fastapi import HTTPException

logger = logging.getLogger(__name__)

RELEASE_DIR = os.environ.get("POLICYLM_RELEASE_DIR", "/app/model_cache/policylm")
# Per request. Each message is scored under the same policy, packed into the helper's batches.
MAX_MESSAGES = int(os.environ.get("POLICYLM_MAX_MESSAGES", "256"))


class Model:
    def __init__(self, **kwargs):
        self._lib = None
        self._model = None

    def load(self):
        # The helper lives in the read-only weights mount; never try to write __pycache__ there.
        sys.dont_write_bytecode = True
        sys.path.insert(0, os.path.join(RELEASE_DIR, "inference"))
        import policylm_infer

        self._lib = policylm_infer
        # dtype "auto" is bfloat16 on CUDA, the read the "balanced" cutoffs were chosen on.
        self._model = policylm_infer.PolicyLM.from_pretrained(RELEASE_DIR, device="cuda")
        self._model.classify("Warm-up message.", "aegis")
        logger.info("PolicyLM loaded: %s", self._model.info)

    def predict(self, model_input):
        if not isinstance(model_input, dict):
            raise HTTPException(status_code=400, detail="request body must be a JSON object")
        unknown = set(model_input) - {"message", "messages", "policy", "cutoff", "clean"}
        if unknown:
            raise HTTPException(status_code=400, detail=f"unknown request keys {sorted(unknown)}")
        if ("message" in model_input) == ("messages" in model_input):
            raise HTTPException(status_code=400, detail="pass exactly one of 'message' or 'messages'")
        messages = [model_input["message"]] if "message" in model_input else model_input["messages"]
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="'messages' must be a non-empty list of strings")
        if len(messages) > MAX_MESSAGES:
            raise HTTPException(
                status_code=400, detail=f"at most {MAX_MESSAGES} messages per request, got {len(messages)}"
            )
        policy = model_input.get("policy", "aegis")

        # The helper validates everything before any forward pass and raises PolicyLMError,
        # TypeError or ValueError for bad input; those are the caller's to fix.
        try:
            report = self._model.check_policy(policy)
            results = self._model.classify_batch(
                messages, policy, cutoff=model_input.get("cutoff"), clean=model_input.get("clean")
            )
        except (self._lib.PolicyLMError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"{type(exc).__name__}: {exc}") from exc

        return {
            "results": [result.to_dict() for result in results],
            "policy_report": {
                "tokens": report.tokens,
                "content_tokens": report.content_tokens,
                "calls": report.calls,
                "notes": list(report.notes),
            },
        }
