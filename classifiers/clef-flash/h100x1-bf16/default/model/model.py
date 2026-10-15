import importlib
import sys

MODEL_NAME = "clef-flash"
REPO_NAME = "Cloudflare/clef-flash"
MODEL_PATH = "/models/clef"
MAX_TOKENS = 16384


class Model:
    def load(self):
        import torch

        sys.path.insert(0, MODEL_PATH)
        self._runtime = importlib.import_module("joint_schema_model")
        self._model, self._processor = self._runtime.load_release_model(
            MODEL_PATH, device="cuda:0", dtype=torch.bfloat16, local_files_only=True,
        )
        self.predict({"model": MODEL_NAME, "state": "The parcel arrived damaged.",
                      "questions": {"damaged": {"type": "noul", "instructions": "Is the parcel damaged?"}}})

    def predict(self, payload):
        # Upstream truncates state silently; refuse oversized inputs before inference.
        encoded = self._runtime.encode_record(
            self._processor.tokenizer, payload, max_length=sys.maxsize, processor=self._processor,
        )
        if len(encoded.input_ids) > MAX_TOKENS:
            raise ValueError(f"request exceeds {MAX_TOKENS} encoded tokens")
        return self._runtime.systemone(self._model, self._processor, payload, max_length=MAX_TOKENS)
