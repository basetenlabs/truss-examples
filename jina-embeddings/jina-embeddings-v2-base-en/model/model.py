from transformers import AutoModel


class Model:
    def __init__(self, **kwargs):
        self.hf_access_token = kwargs["secrets"]["hf_access_token"]
        self._model = None

    def load(self):
        self._model = AutoModel.from_pretrained(
            "jinaai/jina-embeddings-v2-base-en",
            revision="0f472a4cde0e6e50067b8259a3a74d1110f4f8d8",
            trust_remote_code=True,
            use_auth_token=self.hf_access_token,
        )  # Version is pinned to prevent malicious code execution

    def predict(self, model_input):
        if "max_length" not in model_input.keys():
            model_input["max_length"] = 8192
        return self._model.encode(
            model_input["text"], max_length=model_input["max_length"]
        )
