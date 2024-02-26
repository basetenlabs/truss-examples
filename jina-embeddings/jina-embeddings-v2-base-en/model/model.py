import os

from transformers import AutoModel


class Model:
    def __init__(self, **kwargs):
        self.hf_access_token = kwargs["secrets"]["hf_access_token"]
        # There seems to be a bug where transfomers doesn't
        # respect the "token" argument, but this environment
        # variable works.
        os.environ["HF_TOKEN"] = self.hf_access_token
        self._model = None

    def load(self):
        self._model = AutoModel.from_pretrained(
            "jinaai/jina-embeddings-v2-base-en",
            trust_remote_code=True,
            token=self.hf_access_token,
        )

    def predict(self, model_input):
        if "max_length" not in model_input.keys():
            model_input["max_length"] = 8192
        return self._model.encode(
            model_input["text"], max_length=model_input["max_length"]
        )
