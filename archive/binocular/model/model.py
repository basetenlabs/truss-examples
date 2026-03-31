import logging

from binoculars import Binoculars

MINIMUM_TOKENS = 64


class Model:
    def __init__(self, **kwargs):
        self._model = None

    def load(self):
        # Load model here and assign to self._model.
        self._model = Binoculars()
        self._tokenizer = self._model.tokenizer

    def count_tokens(self, text):
        return len(self._tokenizer(text).input_ids)

    def predict(self, model_input: dict):
        input_text = model_input.pop("text")
        if self.count_tokens(input_text) < MINIMUM_TOKENS:
            logging.warn("Insufficient content length")
            return {}

        return {
            "score": self._model.compute_score(input_text),
            "label": self._model.predict(input_text),
        }
