# In this example, we go through building your first Truss model. We'll be using the HuggingFace transformers
# library to build a text classification model that can detect sentiment of text.
# 
# # Step 1: Implementing the model
# 
# Set up imports for this model. In this example, we simply use the HuggingFace transformers library.
from transformers import pipeline

# Every Truss model must implement a `Model` class. This class must have:
#  * an `__init__` function
#  * a `load` function
#  * a `predict` function
#
# In the `__init__` function, set up any variables that will be used in the `load` and `predict` functions.
class Model:
    def __init__(self, **kwargs):
        self._model = None

    # In the `load` function of the Truss, we implement logic
    # involved in downloading the model and loading it into memory.
    # For this Truss example, we define a HuggingFace pipeline, and choose
    # the `text-classification` task, which uses BERT for text classification under the hood.
    # 
    # Note that the the load function runs when the 
    def load(self):
        self._model = pipeline("text-classification")

    # In the `predict` function of the Truss, we implement logic related
    # to actual inference. For this example,  we just call the HuggingFace pipeline
    # that we set up in the `load` function.
    def predict(self, model_input):
        return self._model(model_input)