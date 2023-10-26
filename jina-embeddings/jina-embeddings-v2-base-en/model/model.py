from transformers import AutoModel


class Model:
    def __init__(self, **kwargs):
        self._model = None

    def load(self):
        self._model = AutoModel.from_pretrained(
            "jinaai/jina-embeddings-v2-base-en",
            revision="084b9f6bca3174b98ca82d59dfc0950214aa36df",
            trust_remote_code=True,
        )  # Version is pinned to prevent malicious code execution

    def predict(self, model_input):
        if "max_length" not in model_input.keys():
            model_input["max_length"] = 8192
        return self._model.encode(
            model_input["text"], max_length=model_input["max_length"]
        )
