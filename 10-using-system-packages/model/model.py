# In this example, we build a Truss with a model that requires specific system packages.
#
# To add system packages to your Truss, you can add a `system_packages` key to your config.yaml file,
# for instance:
# To add system packages to your model serving environment, open config.yaml and
# update the system_packages key with a list of apt-installable Debian packages:
#
# ```yaml config.yaml
# system_packages:
#  - tesseract-ocr
# ```
#
# For this example, we use the [LayoutLM Document QA](https://huggingface.co/impira/layoutlm-document-qa) model,
# a multimodal model that answers questions about provided invoice documents. This model requires a system
# package, tesseract-ocr, which needs to be included in the model serving environment.
#
# # Setting up the model.py
#
# For this model, we use the HuggingFace transformers library, and the the document-question-answering task.
from transformers import pipeline


class Model:
    def __init__(self, **kwargs) -> None:
        self._model = None

    def load(self):
        self._model = pipeline(
            "document-question-answering",
            model="impira/layoutlm-document-qa",
        )

    def predict(self, model_input):
        return self._model(
            model_input["url"],
            model_input["prompt"]
        )