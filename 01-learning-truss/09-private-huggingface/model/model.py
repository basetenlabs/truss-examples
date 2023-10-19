# In this example, we build a Truss that uses a model that
# requires Hugging Face authentication. The steps for loading a model
# from Hugging Face are:
#
# 1. Create an [access token](https://huggingface.co/settings/tokens) on your Hugging Face account.
# 2. Add the `hf_access_token`` key to your config.yaml secrets and value to your [Baseten account](https://app.baseten.co/settings/secrets).
# 3. Add `use_auth_token` when creating the actual model.
#
# # Setting up the model
#
# In this example, we use a private version of the [BERT base model](https://huggingface.co/bert-base-uncased). 
# The model is publicly available, but for the purposes of our example, we copied it into a private
# model repository, with the path "baseten/docs-example-gated-model".
#
# First, like with other Hugging Face models, start by importing the `pipeline` function from the
# transformers library, and defining the `Model` class.
from transformers import pipeline

class Model:
    # An important step in loading a model that requires authentication is to
    # have access to the secrets defined for this model. We pull these out of 
    # the keyword args in the `__init__` function.
    def __init__(self, **kwargs) -> None:
        self._secrets = kwargs["secrets"]
        self._model = None

    def load(self):
        # Ensure that when you define the `pipeline`, we use the `use_auth_token` parameter,
        # pass the `hf_access_token` secret that is on our Baseten account.
        self._model = pipeline(
            "fill-mask",
            model="baseten/docs-example-gated-model",
            use_auth_token=self._secrets["hf_access_token"]
        )

    def predict(self, model_input):
        return self._model(model_input)