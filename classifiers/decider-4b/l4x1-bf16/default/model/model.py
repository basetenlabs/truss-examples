import importlib
import sys

MODEL_NAME = "decider-4b"
REPO_NAME = "Mapika/decider-4b"
MODEL_PATH = "/models/decider"
MAX_TOKENS = 32768


class Model:
    def load(self):
        import torch

        sys.path.insert(0, MODEL_PATH)
        runtime = importlib.import_module("decider.infer")
        self._systemone = importlib.import_module("decider.systemone")
        self._model = runtime.Decider(MODEL_PATH, device="cuda:0", dtype=torch.bfloat16, use_graphs=False)
        self.predict({"model": MODEL_NAME, "state": "The parcel arrived damaged.",
                      "questions": {"damaged": {"type": "noul", "instructions": "Is the parcel damaged?"}}})

    def predict(self, payload):
        # The pinned state-first runtime includes this prefix in its context cap.
        context = "Context:\n" + self._systemone.render_state(payload["state"])
        if len(self._model.m.tok.encode(context, add_special_tokens=False)) > MAX_TOKENS:
            raise ValueError(f"context exceeds {MAX_TOKENS} tokens")
        result = self._model.system_one(state=payload["state"], questions=payload["questions"],
                                        max_state_tokens=MAX_TOKENS)
        return {**result, "model": payload["model"]}
