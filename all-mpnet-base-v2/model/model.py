from sentence_transformers import SentenceTransformer


class Model:
    def __init__(self, **kwargs):
        self._model = None

    def load(self):
        self._model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    def predict(self, model_input):
        return self._model.encode(model_input["text"])
