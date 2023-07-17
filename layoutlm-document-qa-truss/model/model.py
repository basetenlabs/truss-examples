from transformers import pipeline


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self._model = None

    def load(self):
        self._model = pipeline(
            "document-question-answering",
            model="impira/layoutlm-document-qa",
        )

    def predict(self, model_input):
        return self._model(
            model_input["url"], # e.g. "https://templates.invoicehome.com/invoice-template-us-neat-750px.png"
            model_input["prompt"] # e.g. "What is the invoice number?"
        )
